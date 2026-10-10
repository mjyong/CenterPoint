# CenterPoint-Pillar 网络详解

对应 [DOG_DATAFLOW.md](DOG_DATAFLOW.md) 4.3 节里的 `net(example, return_loss=False)`。

**模型**：配置为 `configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py`，经 `configure_for_dog` 把范围改为 ±40.8 m；权重为 `work_dirs/pretrained/nusc_pp.pth`。

**测量方式**：在每一层挂 forward hook，记录输出 shape、参数量和乘加次数（MAC）。用的是一帧仿真点云，含 9 663 个非空柱。PFN 的计算量与柱数成正比；样例 bag 那一帧只有 4 607 个柱，PFN 计算量约为表中的一半，其余部分不变。

**坐标系**：网络内部全部在虚拟 nuScenes 雷达系（M 系）：车头朝 +y，地面在 z = −1.84。

## 0. 总览

```
example: voxels (V,20,5)  coordinates (V,4)=[b,z,y,x]  num_points (V,)  shape [408,408,1]
   │
   ▼ reader  PillarFeatureNet          点特征装饰到 10 维 → PFN(10→32, 拼接) → PFN(64→64) → 柱内 max
(V,64)
   ▼ backbone  PointPillarsScatter      按 (y, x) 写进稠密画布，没有参数
(1,64,408,408)            0.2 m/格
   ▼ neck  RPN                          3 级下采样（步长 2/4/8）→ 统一到步长 4 → 拼接
(1,384,102,102)           0.8 m/格
   ▼ bbox_head.shared_conv              3×3 卷积 384→64
(1,64,102,102)
   ▼ 6 个 SepHead（每组 6 个分支）      reg 2 · height 1 · dim 3 · rot 2 · vel 2 · hm C_t
6 × {reg (1,2,·,·), height (1,1), dim (1,3), rot (1,2), vel (1,2), hm (1,C_t)}，空间尺寸均为 102×102
   ▼ bbox_head.predict                  sigmoid / exp / atan2 → 换算成米 → 阈值过滤 → circle NMS
box3d_lidar (D,9)   scores (D,)   label_preds (D,)，D ≤ 6 × 83
```

| 模块 | 参数量 | 计算量 (GMAC) | 占比 |
|---|---|---|---|
| reader（PFN） | 0.005 M | 0.85（9 663 柱时） | 2% |
| backbone（Scatter） | 0 | 0 | 0 |
| neck（RPN） | 4.389 M | 23.9 | 58% |
| head.shared_conv | 0.221 M | 2.3 | 6% |
| head.tasks（6 组 × 6 个分支） | 1.374 M | 14.2 | 35% |
| **合计** | **5.99 M** | **41.3** | |

## 1. 入口：`PointPillars.forward`（`det3d/models/detectors/point_pillars.py:32`）

```python
data = dict(features=voxels, num_voxels=num_points, coors=coordinates,
            batch_size=len(num_voxels), input_shape=example["shape"][0])   # [408, 408, 1]
x = self.extract_feat(data)            # reader → backbone → neck            :21
preds, _ = self.bbox_head(x)           # shared_conv + 6 个 SepHead
return self.bbox_head.predict(example, preds, self.test_cfg)               # return_loss=False 时
```

训练时走同一条前向，只是最后一步换成 `bbox_head.loss`。打开双翻转 TTA 时 `batch_size=4`。

## 2. reader：`PillarFeatureNet`（`readers/pillar_encoder.py:116`）

**输入**：每个柱最多 20 个点，每点 5 维 `[x, y, z, intensity, dt]`；不足 20 个点的位置补 0，真实点数记在 `num_points` 里。

**第 1 步：特征装饰，5 维扩到 10 维。**

| 新增特征 | 计算 | 作用 |
|---|---|---|
| `f_cluster` (3) | `xyz − 本柱内真实点的均值` | 描述柱内点的形状：垂直分布、细长或成团 |
| `f_center` (2) | `x − (col·0.2 + 0.1 − 40.8)`，`y` 同理 | 点在柱内的亚格位置 |
| 原始 5 维 | 绝对 xyz、强度、时间差 | 绝对高度（离地多高）、材质、运动拖影（多帧的 dt） |

装饰完成后，补零的位置要再乘一次掩码，否则它们的 `f_cluster` 会变成 "0 − 均值"，不再是 0。

**第 2 步：两层 PFN。** 这就是配置名里 "two_pfn" 的含义，结构是 PointNet 式的"逐点 MLP + 对称池化"：

| 层 | 运算 | shape |
|---|---|---|
| PFN-1（非最后一层） | Linear 10→32（无 bias）+ BN1d + ReLU | (V,20,10) → (V,20,32) |
| | 20 个点上取 max，得到柱级特征，复制回每个点后拼接 | (V,20,32) ‖ (V,20,32) → (V,20,64) |
| PFN-2（最后一层） | Linear 64→64 + BN1d + ReLU，再对 20 个点取 max | (V,20,64) → (V,64) |

第二层的输入既有逐点特征，也有整柱的全局特征，所以能学到"这个点相对整柱处在什么位置"。这一部分参数只有 5k，计算量正比于 柱数 × 20。

导出给 TensorRT 时（`pillar_export.PFNExport`）的改写：
- Linear 改成 1×1 Conv2d，作用在 `(1,10,P_max,20)` 上；
- BN1d 改成 BN2d；
- 取 max 改成 `max_pool2d(kernel=(1,20))`；
- 输出为 `(1,64,P_max,1)`，已验证与原网络等价（样例帧上相对误差 2e-5）。

这样改是为了得到静态形状、只含稠密算子的图：柱数 V 每帧都在变，补齐到固定的 P_max 之后，TensorRT 引擎就不需要动态形状。

## 3. backbone：`PointPillarsScatter`（`readers/pillar_encoder.py:182`）

```python
canvas = zeros(64, ny*nx)                    # 408*408
indices = coors[:, 2] * nx + coors[:, 3]     # y * nx + x
canvas[:, indices] = voxel_features.T
canvas.view(B, 64, ny, nx)                   # (1, 64, 408, 408)：高度方向是 y，宽度方向是 x
```

这一步没有参数，只做"索引写入"，把稀疏的柱子铺成一张 0.2 m/格的伪图像，空柱的位置全是 0。从这里开始就是纯 2D CNN，所以后面整段都能做成一个静态形状的 TensorRT 引擎。scatter 本身依赖数据决定写入位置，所以部署时放在引擎外面做（`scatter_pillars`，参考实现是 numpy，上板应换成 CUDA）。

## 4. neck：`RPN`（`necks/rpn.py`）

**配置**：`layer_nums=[3,5,5]`、`ds_layer_strides=[2,2,2]`、`ds_num_filters=[64,128,256]`、`us_layer_strides=[0.5,1,2]`、`us_num_filters=[128,128,128]`。

每个 block（`_make_layer`）的结构：`ZeroPad(1) + Conv3×3 stride 2 + BN + ReLU`，后面再接 `layer_nums[i]` 个 `Conv3×3 + BN + ReLU`。所有 BN 的 `eps=1e-3`、`momentum=0.01`。

| 分支 | 结构 | 输入 → 输出 | 相对 BEV 步长 | 每格大小 | GMAC |
|---|---|---|---|---|---|
| `blocks[0]` | Conv s2 64→64 + 3×Conv 64 | (1,64,408,408) → (1,64,204,204) | 2 | 0.4 m | 6.1 |
| `blocks[1]` | Conv s2 64→128 + 5×Conv 128 | → (1,128,102,102) | 4 | 0.8 m | 8.4 |
| `blocks[2]` | Conv s2 128→256 + 5×Conv 256 | → (1,256,51,51) | 8 | 1.6 m | 8.4 |
| `deblocks[0]` | Conv2d k2 s2 64→128（上采样倍率 0.5，实际是下采样） | (1,64,204,204) → (1,128,102,102) | 4 | | 0.34 |
| `deblocks[1]` | Conv2d k1 s1 128→128 | (1,128,102,102) → 同尺寸 | 4 | | 0.17 |
| `deblocks[2]` | ConvTranspose2d k2 s2 256→128 | (1,256,51,51) → (1,128,102,102) | 4 | | 0.34 |
| 拼接 | `torch.cat(ups, dim=1)` | **(1,384,102,102)** | 4 | 0.8 m | |

这是一个类 FPN 的多尺度融合。三条分支依次串联下采样，再各自变换到步长 4，然后拼接：
- 浅层分支（步长 2）保留细节，对应小目标。行人约 0.6 m，只占 3 个柱子。
- 深层分支（步长 8）提供大感受野。最深分支的理论感受野估算约 170 个柱子，约 34 m，足以覆盖 12 m 长的大巴，也能看到周围的上下文。

整个网络的输出步长 `out_size_factor = prod(ds 步长) / us 最后一项 × backbone 的下采样倍数 = 8 / 2 × 1 = 4`，所以 heatmap 每格 0.8 m。网格边长必须能被 8 整除（408 / 8 = 51），这就是检测范围取 ±40.8 m 的原因。

RPN 占了 73% 的参数和 58% 的计算量。仅 `blocks[0]` 在 204×204 上的 4 层卷积就有 6.1 GMAC。

## 5. bbox_head.forward：`CenterHead`（`bbox_heads/center_head.py:236`）

**shared_conv**：`Conv3×3 384→64（有 bias）+ BN + ReLU`，输出 (1,64,102,102)。

**6 个 task**：按尺寸相近把 nuScenes 的 10 类分成 6 组，每组共用一张 heatmap 和同一个 NMS 半径：

| task | 类别 | `hm` 通道数 C_t | circle NMS 半径（配置里是平方米） |
|---|---|---|---|
| 0 | car | 1 | 4 → 2.0 m |
| 1 | truck, construction_vehicle | 2 | 12 → 3.5 m |
| 2 | bus, trailer | 2 | 10 → 3.2 m |
| 3 | barrier | 1 | 1 → 1.0 m |
| 4 | motorcycle, bicycle | 2 | 0.85 → 0.92 m |
| 5 | pedestrian, traffic_cone | 2 | 0.175 → 0.42 m |

**每个 SepHead** 有 6 个独立分支，每个分支都是 `Conv3×3 64→64 + BN + ReLU + Conv3×3 64→C`（`num_conv=2`、`head_conv=64`、`final_kernel=3`）：

| 分支 | 通道 | 含义（训练目标） | 解码 |
|---|---|---|---|
| `reg` | 2 | 中心点在 heatmap 格子内的亚格偏移 `ct − floor(ct)`，范围 [0,1) | 加到格子坐标上 |
| `height` | 1 | 框中心的绝对 z（M 系） | 直接使用 |
| `dim` | 3 | `log(w, l, h)` | `exp` |
| `rot` | 2 | `(sin r, cos r)`，其中 `r = −yaw − π/2` | `atan2(sin, cos)` |
| `vel` | 2 | 对地速度 `(vx, vy)`，单位 m/s（由多帧拖影和 dt 学到） | 直接使用 |
| `hm` | C_t | 每类一张中心点热力图的 logit | `sigmoid` |

- `hm` 最后一层的 bias 初始化为 −2.19，即 sigmoid ≈ 0.1，这是 focal loss 训练稳定的常用技巧。训练后的值，例如 task 5 为 `[-0.96, -1.08]`。
- **计算量提示**：36 个分支各自的第一层 3×3 卷积（64→64，102×102）各约 0.38 GMAC，合计约 14 GMAC，占全网 35%。机器狗的 3 类配置只有 3 个 task、18 个分支，这部分计算量减半，参数为 5.30 M。若要进一步压缩，可以把 `head_conv` 降到 32，或让同一 task 的各分支共用第一层卷积。

**训练目标**（`AssignLabel`，帮助理解各通道的含义）：
- 每个 GT 在其中心所在格子画一个高斯，半径由框的尺寸决定（`gaussian_overlap=0.1`，最小 2 格），用 FastFocalLoss 监督 heatmap；
- 回归项只在 GT 中心格计算，目标向量为 10 维 `[reg 2, z 1, log dim 3, vel 2, sin 1, cos 1]`，用 L1 loss，`code_weights` 中速度两项权重为 0.2；
- 每个 task 的损失为 `hm_loss + 0.25 × loc_loss`，6 个 task 相加。

## 6. predict + post_processing（`center_head.py:294` / `:451`）

对每个 task：

```python
pred = pred.permute(0, 2, 3, 1)                     # NCHW → NHWC
hm   = sigmoid(hm)                                  # (1,102,102,C_t)
dim  = exp(dim)
r    = atan2(rot[..., 0], rot[..., 1])
xs   = col + reg_x ; ys = row + reg_y               # 格子坐标 + 亚格偏移
x = xs * 4 * 0.2 + (-40.8) ; y = ys * 4 * 0.2 + (-40.8)   # 换算成米（M 系）
box = [x, y, height, w, l, h, vx, vy, r]            # 102×102 = 10 404 个候选，每个 9 维
# post_processing
score, label = hm.max(-1), hm.argmax(-1)
keep = (score > 0.1) & (中心在 post_center_limit_range 内)
keep = circle_nms(按中心距离, 半径为本 task 的 min_radius)[:83]
```

举例：task 5 的 pedestrian 通道在 `(row=60, col=40)` 处峰值为 0.7，`reg=(0.3, 0.7)`，则
`x = (40 + 0.3) × 0.8 − 40.8 = −8.56 m`，`y = (60 + 0.7) × 0.8 − 40.8 = 7.76 m`（M 系）。

最后把 6 个 task 的结果拼接起来，label 加上前面各 task 的类别数偏移，得到全局类别 id 0–9。输出为 `box3d_lidar (D,9)`、`scores (D,)`、`label_preds (D,)`，D 最多 6 × 83 = 498。

需要注意的几点：
- **没有 CenterNet 那种 3×3 max-pool 取峰值。** 一个峰值周围的相邻格子分数也会超过阈值，全靠 circle NMS 按半径去重。所以 NMS 半径和分数阈值直接决定框的数量和重复程度，这也是上次"框太多"要在输出端加过滤的原因之一。
- 不同 task 之间不做 NMS，所以 car 和 truck 两组可能在同一辆车上各出一个框。我们的 `postprocess` 在映射到 3 类之后，会再做一次类内合并（`classwise_circle_nms`）。
- 打开双翻转 TTA 时，`predict` 先把 4 份结果翻回原方向，对 hm、reg、dim、rot、vel 取平均，再解码。

## 7. 与部署（Orin / TensorRT）的对应关系

| 部分 | 运行位置 | 实现 |
|---|---|---|
| 体素化、点特征装饰、补齐到固定柱数 | 主机 → 上板改 GPU | `VoxelGenerator`、`decorate_pillars`、`pad_pillars` |
| PFN | TensorRT | `pfn.onnx`：(1,10,30000,20) → (1,64,30000,1) |
| Scatter | 主机 → 上板改 GPU | `scatter_pillars` → (1,64,408,408) |
| RPN + shared_conv + SepHead | TensorRT | `rpn_head.onnx`：只有 Conv、BN、ReLU、ConvTranspose、Concat；输出 36 张图 |
| 解码 + circle NMS | 主机（阈值和 top-k 可放 GPU） | `decode.py`，与 `CenterHead.predict` 逐框一致（有单测） |

RPN 每个下采样块开头的 `ZeroPad2d(1) + Conv(padding=0)` 在导出时被合并成 `Conv(padding=1)`，结果完全相同，但 ONNX 里不再有运行时计算 pads 的 Pad 子图，老版本 TensorRT 解析器也能接受。完整的上板步骤见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节。
