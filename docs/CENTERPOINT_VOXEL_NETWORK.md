# CenterPoint-Voxel 网络详解

对应 [DOG_DATAFLOW.md](DOG_DATAFLOW.md) 4.3 节里 Voxel 分支的 `net(example, return_loss=False)`。Pillar 版见 [CENTERPOINT_NETWORK.md](CENTERPOINT_NETWORK.md)，两者共用的部分（head 各分支的含义、训练目标、解码细节）在那里讲得更细，本文只写 Voxel 不同的地方。第 9 节是与 Pillar 的逐项对比。

**模型**：配置为 `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`，经 `configure_for_dog` 把 xy 范围改为 ±40.8 m（z 不动，见 3.4 节）；权重为 `work_dirs/pretrained/nusc_voxel.pth`（Git LFS，103 MB）。机器狗的微调配置 `configs/dog/dog_centerpoint_voxel_xt32.py` 的网络完全相同，只是 head 从 6 个 task 变成 5 个（5.3 节）。

**测量方式**：`tools/dog/profile_detector.py`，在每一层挂 forward hook，记录输出 shape、非空格数、参数量和乘加次数（MAC）。用的是样例 bag `rec_moving` 的一帧（`1789980192.298542.npy`，5 帧叠加共 187 434 点，进入范围的 148 734 点）。同一帧给 Pillar 是 9 658 个非空柱。
- 稀疏卷积的计算量按实际的"（输入格，输出格，卷积核偏移）三元组数 × C_in × C_out"统计，所以主干部分随帧变化；稠密部分与帧无关。
- 这次是在 CPU 上测的。spconv 2.x 的 CPU 内核不接受融合在卷积里的 bias（`cpu don't support act and bias`），脚本把稀疏卷积切到 train 模式，让 bias 在 gemm 之后单独加上，结果相同，BN 仍是 eval。另外，spconv 在 CPU 上多线程时结果不确定（同一帧两次前向，heatmap 概率最多差 0.13），所以脚本在 CPU 上只用单线程。
- 在 4090 上加 `--device cuda:0` 可以测出各段的 GPU 耗时。

```bash
python tools/dog/profile_detector.py --ckpt work_dirs/pretrained/nusc_voxel.pth \
    --frame data/runs/rec_moving/frames/1789980192.298542.npy --layers [--device cuda:0] [--tta]
```

**坐标系**：网络内部全部在虚拟 nuScenes 雷达系（M 系）：车头朝 +y，地面在 z = −1.84。

## 0. 总览

```
example: voxels (V,10,5)  coordinates (V,4)=[b,z,y,x]  num_points (V,)  shape [1088,1088,40]
   │
   ▼ reader  VoxelFeatureExtractorV3      体素内的点取均值，没有参数
(V,5)                                     本帧 V = 34 934
   ▼ backbone  SpMiddleResNetFHD          稀疏 3D ResNet：只在非空体素上算；xy 下采样 8 倍，z 41 → 2
稀疏张量 6 726 个非空格 × 128 通道，网格 (z,y,x) = 2×136×136
   ▼ .dense() + view                      两层高度并进通道：128 × 2 = 256
(1,256,136,136)           0.6 m/格
   ▼ neck  RPN                            2 级（步长 1 / 2）→ 统一到 136×136 → 拼接
(1,512,136,136)           0.6 m/格
   ▼ bbox_head.shared_conv                3×3 卷积 512→64
(1,64,136,136)
   ▼ 6 个 SepHead（每组 6 个分支）        与 Pillar 相同：reg 2 · height 1 · dim 3 · rot 2 · vel 2 · hm C_t
6 × {reg (1,2,·,·), height (1,1), dim (1,3), rot (1,2), vel (1,2), hm (1,C_t)}，空间尺寸均为 136×136
   ▼ bbox_head.predict                    公式与 Pillar 相同，只是每格 0.6 m → 阈值过滤 → circle NMS
box3d_lidar (D,9)   scores (D,)   label_preds (D,)，D ≤ 6 × 83
```

| 模块 | 参数量 | 计算量 (GMAC) | 占比 |
|---|---|---|---|
| reader（取均值） | 0 | ≈0 | 0 |
| backbone（稀疏 3D 卷积） | 2.695 M | 13.5（34 934 个体素时） | 17% |
| neck（RPN） | 4.577 M | 35.9 | 45% |
| head.shared_conv | 0.295 M | 5.5 | 7% |
| head.tasks（6 组 × 6 个分支） | 1.374 M | 25.3 | 32% |
| **合计** | **8.94 M** | **80.2** | |

与 Pillar（5.99 M、41.3 GMAC）相比，参数多 50%，计算量约为 2 倍。增加的 39 GMAC 里约 2/3 来自 136×136 上的稠密部分（neck 和 head），稀疏主干只占 1/3。

## 1. 入口：`VoxelNet.forward`（`det3d/models/detectors/voxelnet.py:55`）

```python
# extract_feat（:23），example 里已有 voxels 时走这一支
data = dict(features=example["voxels"], num_voxels=example["num_points"], coors=example["coordinates"],
            batch_size=len(example["points"]), input_shape=example["shape"][0])     # [1088, 1088, 40]
input_features = self.reader(data["features"], data["num_voxels"])                  # (V,5)
x, voxel_feature = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
x = self.neck(x)
# forward（:55）
preds, _ = self.bbox_head(x)
return self.bbox_head.predict(example, preds, self.test_cfg)                         # return_loss=False 时
```

- 和 `PointPillars.forward` 的结构完全一样，只是 reader 和 backbone 换了。训练时最后一步换成 `bbox_head.loss`。打开双翻转 TTA 时 `batch_size=4`。
- `voxel_feature` 是主干各级的稀疏特征（conv1–conv4），只给两阶段的 `forward_two_stage` 用。我们不用两阶段：它的 ROI head 依赖 `iou3d_nms` CUDA 算子，我们的环境里没有编译（启动时会打印 `iou3d_nms not built`）。
- `extract_feat` 还有一支：example 里没有 `voxels` 时，由 reader 自己从原始点做动态体素化。我们的配置都在数据管道里体素化，这一支不会走到。

## 2. 体素化与 reader

### 2.1 体素化（`VoxelGenerator`，`det3d/core/input/voxel_generator.py`）

| 参数 | 值 | 说明 |
|---|---|---|
| `voxel_size` | 0.075 × 0.075 × 0.2 m | 水平方向比 Pillar 的 0.2 m 细 2.7 倍（面积细 7.1 倍），高度方向分成 0.2 m 一层 |
| `range` | x、y 为 ±40.8 m，z 为 [−5, 3]（M 系） | z 不能随意改，见 3.4 节 |
| 网格 (x, y, z) | 1088 × 1088 × 40 | |
| `max_points_in_voxel` | 10 | 超出的点直接丢掉，先到先得 |
| `max_voxel_num` | 训练 120 000 / 推理 160 000 | 本帧只用了 34 934 个 |

**输出**：`voxels (V,10,5)`，每点 `[x, y, z, intensity, dt]`，不足 10 个点的位置补 0；`coordinates (V,3)`，顺序为 (z, y, x)，封装里在最前面补一列 batch 号，变成 `(V,4)`；`num_points (V,)` 是每个体素里的真实点数。

**样例帧的统计**：

| | Voxel | Pillar（同一帧） |
|---|---|---|
| 非空格数 | 34 934 个体素 | 9 658 个柱 |
| 每格平均点数 | 3.16 | 7.72 |
| 装满（点被丢弃）的格 | 8.8%（满 10 点） | 16.7%（满 20 点） |
| 网格占用率 | 0.072%（34 934 / 41×1088×1088） | 5.8%（9 658 / 408×408） |

- 34 934 个体素落在 30 932 个不同的 (y, x) 位置上，平均每个位置只有 1.13 层，绝大多数是一层地面。
- 机器狗管线先在检测系里裁剪 z ∈ [−1, 5]，换到 M 系后是 [−2.84, 3.16]，上端再被网格截在 3（离地 4.84 m）。所以 40 层里最下面 10 层（第 0–9 层）永远是空的，本帧占用的是第 12–39 层。
- 地面（z ≈ −1.84，第 15、16 层）一共 18 658 个体素，占全部的 53%。

### 2.2 reader：`VoxelFeatureExtractorV3`（`readers/voxel_encoder.py:9`）

```python
points_mean = features[:, :, :5].sum(dim=1) / num_voxels.view(-1, 1)   # num_voxels 实为每个体素的真实点数
return points_mean                                                     # (V,5)，补零的位置加的是 0，不影响均值
```

- 没有参数，输出 `(V,5)`：体素内点的平均 `[x, y, z, intensity, dt]`。
- 均值 xyz 是点的质心，不是格子中心，所以保留了亚体素位置。这和 Pillar 的 `f_center` 起的作用一样，只是这里不用学。
- 为什么取均值就够：一个 7.5 cm × 7.5 cm × 20 cm 的体素平均只有 3 个点，取均值几乎不丢信息。Pillar 不同：要把一根 8 m 高、最多 20 个点的柱子压成一个向量，只能靠 PFN 去学。Voxel 的"点特征学习"全部推迟到稀疏 3D 卷积里做。
- 多帧叠加时，`dt` 也被平均。同一个体素里如果混有不同时刻的点（运动物体的拖影），均值 dt 介于两者之间，网络从相邻体素的 dt 变化里学速度。

## 3. backbone：`SpMiddleResNetFHD`（`backbones/scn.py:98`）

### 3.1 两种稀疏卷积

| 算子 | 输出哪些格 | 用途 |
|---|---|---|
| `SubMConv3d`（子流形卷积） | 只输出输入里已有的非空格；求和时只取非空的邻居 | 各阶段内部的特征提取。非空集合不"膨胀"，层数再多也不会糊成一片 |
| `SparseConv3d`（普通稀疏卷积，步长 2） | 只要卷积窗里有一个非空输入，输出格就非空 | 下采样 |

- 计算量 = 非空的（输入, 输出, 偏移）对数 × C_in × C_out。如果把 conv1 的一层 16→16 的 3×3×3 卷积做成稠密的，要在 41×1088×1088 = 4 853 万个格子上算，需要 335 GMAC；稀疏版只要 0.05 GMAC。
- 同一阶段的 SubM 层用同一个 `indice_key`（`res0`–`res3`）共享一张邻接表，每个阶段只建一次。

### 3.2 稀疏网格的形状

```python
sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]       # [40,1088,1088] + [1,0,0] = [41,1088,1088]，顺序 (z,y,x)
ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
```

z 方向多加的 1 层永远是空的，因为体素化只有 40 层。加这一层是为了让 z 方向的下采样正好走到 2 层：41 → 21 → 11 → 5 → 2。

### 3.3 逐层结构（本帧实测）

`SparseBasicBlock` 的结构：`SubM 3³（有 bias）+ BN1d + ReLU + SubM 3³（有 bias）+ BN1d`，再加残差，最后 ReLU。BN 的 `eps=1e-3`、`momentum=0.01`。卷积后面紧跟 BN，所以卷积的 bias 其实多余，但预训练权重里有它。CPU 版 spconv 报错就是因为这些 bias。

| 阶段 | 结构 | 通道 | 网格 z×y×x | 每格 xy / z | 非空格 | SubM 平均邻居（满 27） | GMAC | 参数 |
|---|---|---|---|---|---|---|---|---|
| 输入 | | 5 | 41×1088×1088 | 0.075 / 0.2 m | 34 934 | | | |
| `conv_input` | SubM 3³ 5→16（无 bias）+ BN + ReLU | 16 | 同上 | 同上 | 34 934 | 5.6 | 0.02 | 2 192 |
| `conv1` | 2 × Block(16) | 16 | 同上 | 同上 | 34 934 | 5.6 | 0.20 | 27 840 |
| `conv2` | SpConv 3³ s2 16→32 + BN + ReLU，2 × Block(32) | 32 | 21×544×544 | 0.15 / 0.4 m | 33 653 | 12.2 | 1.75 | 124 864 |
| `conv3` | SpConv 3³ s2 32→64 + BN + ReLU，2 × Block(64) | 64 | 11×272×272 | 0.3 / 0.8 m | 16 491 | 13.9 | 3.99 | 498 560 |
| `conv4` | SpConv 3³ s2 64→128，z 方向不补边（`padding=[0,1,1]`）+ BN + ReLU，2 × Block(128) | 128 | 5×136×136 | 0.6 / 1.6 m | 7 237 | 14.6 | 7.40 | 1 992 448 |
| `extra_conv` | SpConv 核 (3,1,1)、步长 (2,1,1)，128→128 + BN + ReLU，只压 z 方向 | 128 | 2×136×136 | 0.6 / 3.2 m | 6 726 | | 0.17 | 49 408 |
| `.dense()` + `view` | `(1,128,2,136,136)` → `(1,256,136,136)` | 256 | 136×136 | 0.6 m | 稠密 | | 0 | 0 |
| **合计** | | | | | | | **13.5** | **2.695 M** |

可以看出：
- **原分辨率下非常稀疏。** 7.5 cm 的体素平均只有 4.6 个非空邻居（5.6 含自己）。点大多落在面上（地面、墙面），一个水平面在 3×3×3 的窗口里只占 9 格。下采样以后，每格的邻居增加到 12–15 个。
- **非空格数**：conv2 虽然把分辨率减半，但 3³ 的窗口会让非空集合向外扩，所以非空格数几乎不变（34 934 → 33 653）。之后每级大约减半：16 491 → 7 237 → 6 726。
- **conv4 是大头。** 它占主干 55% 的计算量（7.40 / 13.5）和 74% 的参数（1.99 M / 2.70 M），因为 128 通道的 3×3×3 卷积参数量是 conv3 的 4 倍。
- **计算量随场景变化。** 主干的计算量与非空体素数大致成正比：点越多、叠加的帧越多，计算量越大；TTA 是 4 倍。后面的稠密部分是固定的。
- **转成稠密后**：2×136×136 = 36 992 个格子里有 6 726 个非空（18%），其余位置填 0。
- **感受野**：主干输出处的理论感受野在 xy 方向约 137 格 × 0.075 m ≈ 10.3 m；z 方向覆盖全部 41 层。

### 3.4 为什么 z 网格必须保持 [−5, 3]

`view(N, C*D, H, W)` 把 D = 2 个高度层并进通道：第 2c 个通道是第 c 个特征在下层的值，第 2c+1 个是在上层的值。这带来两条约束：

1. **形状约束。** neck 的第一层卷积要求 256 个输入通道，所以主干出来必须是 D = 2。按各层的步长和补边推算，z 网格为 40–55 格时 D = 2；超出这个范围，例如把 z 改成 [−3, 3]（30 格，31 → 16 → 8 → 3 → 1），只剩 128 个通道，前向时 neck 第一层直接报通道数不匹配（权重本身能加载，因为卷积参数的形状与 z 无关）。
2. **语义约束。** 即使格数在 40–55 之间，预训练的 neck 也已经学会了"哪些通道代表哪个高度"。往回推卷积窗口的中心：下层对应输入的第 12 层附近（z ≈ −2.5 m，地面以下约 0.7 m），上层对应第 28 层附近（z ≈ 0.7 m，地面以上约 2.5 m）。两层的感受野都覆盖全部高度，但侧重不同。如果平移 z 范围，物体会落到另一组通道上，形状对得上，精度却会掉。

所以 `configure_for_dog` 只改 xy，不动 z；高度方向的过滤在检测系里裁剪点云来做（2.1 节）。Pillar 没有这个问题：它只有一个 z 柱，z 范围只起裁剪作用。

## 4. neck：`RPN`（`necks/rpn.py`）

**配置**：`layer_nums=[5,5]`、`ds_layer_strides=[1,2]`、`ds_num_filters=[128,256]`、`us_layer_strides=[1,2]`、`us_num_filters=[256,256]`、`num_input_features=256`。block 的结构与 Pillar 相同：`ZeroPad(1) + Conv3×3 + BN + ReLU`，再接 `layer_nums[i]` 个 `Conv3×3 + BN + ReLU`；两条分支串联下采样，再各自变换回 136×136 后拼接。

| 分支 | 结构 | 输入 → 输出 | 相对 BEV 步长 | 每格大小 | GMAC | 参数 |
|---|---|---|---|---|---|---|
| `blocks[0]` | Conv **s1** 256→128 + 5×Conv 128 | (1,256,136,136) → (1,128,136,136) | 1 | 0.6 m | 19.1 | 1.034 M |
| `blocks[1]` | Conv s2 128→256 + 5×Conv 256 | → (1,256,68,68) | 2 | 1.2 m | 15.0 | 3.247 M |
| `deblocks[0]` | Conv2d k1 s1 128→256 + BN + ReLU | (1,128,136,136) → (1,256,136,136) | 1 | | 0.61 | 0.033 M |
| `deblocks[1]` | ConvTranspose2d k2 s2 256→256 + BN + ReLU | (1,256,68,68) → (1,256,136,136) | 1 | | 1.21 | 0.263 M |
| 拼接 | `torch.cat(ups, dim=1)` | **(1,512,136,136)** | 1 | 0.6 m | | |

与 Pillar 的 RPN（3 级，第一级就 s2 下采样）相比：
- 稀疏主干已经下采样了 8 倍，所以 RPN 第一级不再下采样，只有两级。
- `blocks[0]` 的 6 层 3×3 卷积全在 136×136 上，单这一块就有 19.1 GMAC，占全网的 24%，是最大的一块。
- 输出步长 `out_size_factor = 主干的 ds_factor × prod(ds 步长) / us 最后一项 = 8 × 2 / 2 = 8`，heatmap 每格 0.075 × 8 = 0.6 m。
- **网格尺寸约束**：xy 格数必须是 主干步长 × RPN 下采样 = 8 × 2 = 16 的整数倍（`_required_divisor`）。1088 = 68 × 16，即范围 2·XY 必须是 0.075 × 16 = 1.2 m 的整数倍。Pillar 要求 2·XY 是 0.2 × 8 = 1.6 m 的整数倍。±40.8 m（81.6 m = 68 × 1.2 = 51 × 1.6）两个都满足，这就是选它的原因。
- **感受野**：浅分支（含 head）约 21 m；深分支约 34 m，和 Pillar 最深分支差不多。

## 5. bbox_head：`CenterHead`（`bbox_heads/center_head.py:236`）

### 5.1 与 Pillar 相同的部分

6 个 task 的分组、每个 SepHead 的 6 个分支（`reg`、`height`、`dim`、`rot`、`vel`、`hm`）、`hm` 的 bias 初始化（−2.19）、每个分支的含义和解码方式都与 Pillar 一样，见 [CENTERPOINT_NETWORK.md](CENTERPOINT_NETWORK.md) 第 5 节。配置里另外写了 `share_conv_channel=64`、`dcn_head=False`，用的是普通卷积 head，不用 DCN 版。

### 5.2 不同的部分

| | Voxel | Pillar |
|---|---|---|
| `shared_conv` | Conv3×3 **512**→64 + BN + ReLU，0.295 M 参数，5.5 GMAC | Conv3×3 384→64，0.221 M，2.3 GMAC |
| head 特征图 | **136×136，0.6 m/格** | 102×102，0.8 m/格 |
| 每个分支第一层 3×3 卷积 | 0.68 GMAC | 0.38 GMAC |
| 36 个分支合计 | 25.3 GMAC（占全网 32%） | 14.2 GMAC（35%） |

head 的参数量两者相同（1.374 M），计算量与特征图面积成正比：136² / 102² = 1.78 倍。

### 5.3 机器狗配置

`configs/dog/dog_centerpoint_voxel_xt32.py` 有 5 个 task（car、truck、bus、pedestrian、cyclist，各 1 类），共 30 个分支，输出时 car / truck / bus 合并为 vehicle（[DOG_FINETUNE.md](DOG_FINETUNE.md) 第 12 节）。实测 tasks 部分 1.14 M 参数、21.0 GMAC；整个网络 **8.71 M 参数、75.9 GMAC**。Pillar 的狗配置是 5.76 M、38.9 GMAC。旧的单 vehicle head 布局（`*_3cls.py`，3 个 task）分别是 8.25 M、67.5 GMAC 和 5.30 M、34.1 GMAC。

### 5.4 训练目标

与 Pillar 相同（高斯热力图 + FastFocalLoss，回归项 L1，`code_weights` 中速度两项为 0.2，每个 task 的损失为 `hm_loss + 0.25 × loc_loss`），只是高斯画在 0.6 m 的格子上：`out_size_factor=8`、`gaussian_overlap=0.1`、最小半径 2 格，即 1.2 m（Pillar 为 1.6 m）。行人中心的定位因此更细。

## 6. predict + post_processing（`center_head.py:294` / `:451`）

代码与 Pillar 完全相同，只是两个参数不同：`out_size_factor=8`、`voxel_size=0.075`。

```python
x = (col + reg_x) * 8 * 0.075 + (-40.8)       # = (col + reg_x) × 0.6 − 40.8（M 系）
y = (row + reg_y) * 8 * 0.075 + (-40.8)
```

举例：task 5 的 pedestrian 通道在 `(row=80, col=54)` 处峰值为 0.7，`reg=(0.3, 0.7)`，则
`x = (54 + 0.3) × 0.6 − 40.8 = −8.22 m`，`y = (80 + 0.7) × 0.6 − 40.8 = 7.62 m`（M 系）。

- **候选数**：每个 task 136 × 136 = 18 496 个，6 个 task 共 110 976 个；Pillar 每个 task 102 × 102 = 10 404 个。
- **NMS**：nuScenes 原配置用的是旋转 IoU NMS（`use_rotate_nms=True`，依赖 `iou3d_nms` CUDA 算子）。`configure_for_dog` 把它换成 circle NMS，每个 task 的半径与 Pillar 相同（`CIRCLE_NMS_SQ_RADIUS`），每个 task 最多保留 83 个，分数阈值 0.1，中心限制在 ±40.8 m 内。之后映射到 3 类、做类内合并（`classwise_circle_nms`），也与 Pillar 相同。
- **双翻转 TTA**：与 Pillar 相同，4 份翻转点云组成 batch=4，`predict` 先翻回再平均。计算量是 4 倍（本帧 320 GMAC）。
- 我们的 numpy 解码器 `decode.py` 从 `test_cfg` 读步长和体素大小，对 Voxel 同样适用。单测 `test_decoder_matches_det3d_predict` 在 Pillar（102×102）和 Voxel（136×136）上都验证了它与 `CenterHead.predict` 逐框一致。

## 7. 训练（机器狗配置）

`dog_centerpoint_voxel_xt32.py` 与 `dog_centerpoint_pp_xt32.py` 只在模型和体素化两处不同（Voxel 的 assigner 里多写的 `dense_reg=1` 在 det3d 里没有被用到）。训练相关的设置完全一样：
- 5 帧叠加；
- GT-AUG：car 2、truck 2、bus 2、pedestrian 4、cyclist 4（GT 库里没有的类别跳过）；
- 全局增强：旋转 ±45°、缩放 0.95–1.05、平移 σ = 0.1 m、俯仰和横滚 ±2°；
- 优化：Adam + one-cycle，`lr_max=5e-4`，20 个 epoch，batch 4；
- 热启动：`tools/dog/convert_nusc_ckpt.py`。car、truck、bus、pedestrian 各自继承 nuScenes 同名的 head，cyclist 继承 bicycle，主干、neck 和 shared_conv 原样拷贝。

需要注意：
- 训练同样需要 CUDA 版 spconv。
- 训练时 `max_voxel_num` 是 120 000。样例帧（5 帧叠加）约 3.5 万个体素，加上 GT-AUG 贴进来的物体，也远不到上限。

具体命令见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)。

## 8. 部署（Orin）

| 部分 | 运行位置 | 实现 / 现状 |
|---|---|---|
| 体素化 + 取均值 | GPU（CUDA kernel） | 本仓库是 numba 实现（`VoxelGenerator`）；取均值是一行运算，可以和体素化合并 |
| 稀疏 3D 主干 | spconv 2.x（PyTorch）或 libspconv（C++） | 非空格数每帧都在变；TensorRT 没有原生的稀疏卷积，进不了静态引擎 |
| dense + RPN + shared_conv + SepHead | TensorRT | 与 Pillar 的 `rpn_head.onnx` 同类，只有 Conv、BN、ReLU、ConvTranspose、Concat，输入固定为 (1,256,136,136)。本仓库还没有 Voxel 的导出脚本，`export_pillar_onnx.py` 只导 Pillar |
| 解码 + circle NMS | 主机 | 与 Pillar 共用 `decode.py` |

上板路线见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8.2 节：
1. 先用 PyTorch + spconv 直接跑，测精度和延迟；
2. 值得的话，再走 NVIDIA Lidar_AI_Solution 的 CUDA-CenterPoint（libspconv + TensorRT）。它针对的正是 nuScenes 的 0.075 m 模型，但我们的网格是 1088（官方 1440），而且是 3 个 task，需要重新导出并修改解码。

显存方面，进入稠密部分的 BEV 特征是 (1,256,136,136)，FP32 下 18.9 MB；RPN 输出 (1,512,136,136) 是 37.9 MB。两者与 Pillar 的 (1,64,408,408) 画布（42.6 MB）在同一量级，显存不是问题。真正的差别在于稀疏主干要用 spconv，工具链也因此更复杂。

## 9. 与 Pillar 的区别

| 项目 | Pillar（`nusc_centerpoint_pp_02voxel_two_pfn_10sweep`） | Voxel（`nusc_centerpoint_voxelnet_0075voxel_fix_bn_z`） |
|---|---|---|
| 输入单元 | 竖直柱 0.2 × 0.2 m，高 8 m，z 不分层 | 体素 0.075 × 0.075 × 0.2 m |
| 网格 (x, y, z) | 408 × 408 × 1 | 1088 × 1088 × 40（稀疏网格 z 为 41） |
| 每格最多点数 | 20 | 10 |
| 样例帧非空格数 | 9 658 个柱，网格占用 5.8% | 34 934 个体素，网格占用 0.072% |
| reader | PFN：点特征装饰到 10 维 → 两层 PFN → 取 max，(V,64)，4.6 k 参数 | 体素内取均值，(V,5)，没有参数 |
| 高度信息怎么进网络 | 柱内各点的 z 经 PFN 学成一个向量 | 保留 0.2 m 一层做 3D 卷积，最后压成上下 2 层并进通道 |
| backbone | Scatter 成稠密伪图像，没有参数 | 稀疏 3D ResNet（SubM + 步长 2 稀疏卷积），2.70 M 参数 |
| 进入 2D 部分的 BEV | (1,64,408,408)，0.2 m/格 | (1,256,136,136)，0.6 m/格 |
| neck | 3 级（s2/s2/s2），输出 (1,384,102,102) | 2 级（s1/s2），输出 (1,512,136,136) |
| head 特征图 | 102 × 102，0.8 m/格 | 136 × 136，0.6 m/格 |
| 输出步长 `out_size_factor` | 4（相对 0.2 m 柱） | 8（相对 0.075 m 体素） |
| xy 范围约束 | 2·XY 是 1.6 m 的整数倍 | 2·XY 是 1.2 m 的整数倍 |
| z 范围 | 可以随意改，只起裁剪作用 | 必须保持 [−5, 3]（3.4 节） |
| 参数量（nuScenes 6 task / 狗 5 task） | 5.99 M / 5.76 M | 8.94 M / 8.71 M |
| 计算量（nuScenes 6 task / 狗 5 task） | 41.3 / 38.9 GMAC，基本固定 | 80.2 / 75.9 GMAC，其中稀疏主干 13.5 随点数变化 |
| 计算量的最大头 | RPN（58%） | RPN（45%），其中 `blocks[0]` 一块就占 24% |
| nuScenes 验证集（官方） | mAP 50.3 / NDS 60.2 | mAP 59.6 / NDS 66.8 |
| 官方测速（TITAN RTX） | 31 FPS | 0.1 m 版 16 FPS，0.075 m 版没有列出（更慢） |
| 4090 上本管线实测（网络部分，中位数） | 129.5 / 133.8 ms（rec_static / rec_moving） | 112.0 / 115.7 ms；加 TTA 122.1 / 126.3 ms |
| CPU | 可以跑（约 0.6 s/帧，多线程） | 封装里不支持（spconv CPU 报错）；只有本文的测量脚本绕过 |
| Orin 部署 | 两个静态形状的 TensorRT 引擎，工具链已就绪（`export_pillar_onnx.py`） | 稀疏主干要 spconv / libspconv，稠密部分可上 TensorRT；导出脚本未写 |
| 在本项目里的角色 | 线上默认检测器 | 离线教师（自动标注）和精度参照；Orin 上实测放得下才考虑上线 |

**怎么理解这些区别：**
- **精度差距主要来自"高度"和"分辨率"。** Pillar 把整根柱子压成一个向量，上下叠在一起的结构（车顶上方的树枝、人头顶上的雨棚）会混在同一个特征里。Voxel 在 3D 里先分开处理，再压成 BEV；输入的水平分辨率又细 2.7 倍，head 每格 0.6 m（Pillar 0.8 m）。行人、自行车这类小目标受益最大，这也是我们用它当教师的原因。
- **计算量的性质不同。** Pillar 的主体是 408×408 上的固定稠密卷积，点多点少几乎不影响耗时，最适合 TensorRT 静态引擎。Voxel 主干的计算量正比于非空体素数，点多、叠加的帧多、开 TTA 都会线性增加，而且它依赖 spconv，在 TensorRT 之外运行。
- **4090 上 Voxel 反而略快，不能说明它更省。** 按 4090 的算力估算，两个模型的卷积都只要几毫秒，测到的 110–130 ms 远超于此。另外，TTA 让计算量变成 4 倍，耗时却只多 10 ms。这说明瓶颈不在卷积，而在框架和数据搬运等其他环节，具体在哪里还没有拆开测过。想弄清楚，在 4090 上运行 `profile_detector.py --device cuda:0`，看各段耗时。Orin 的算力小一个数量级，卷积本身会变成主要开销，到时 Voxel 的 2 倍计算量和 spconv 的开销都会体现出来，必须上板实测。
- **两者共用的部分**：输入点的格式（M 系的 `[x, y, z, intensity, dt]`）、6 个 task 的 head 结构和各分支含义、解码公式、circle NMS、TTA、训练目标和损失、我们的后处理与过滤。两个模型能并行跑同一份预处理输出（`run_sequence.py --detectors pillar,voxel`），对比方法见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 中"Pillar 与 Voxel 的对比方法"一段。
