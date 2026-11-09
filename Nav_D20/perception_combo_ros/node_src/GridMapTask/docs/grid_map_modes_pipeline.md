# GridMapTask 处理流程说明

> 对应节点：`grid_map_memory_node`  
> 配置：`cfg/grid_map_memory_node.yaml`（运行时加载 `install/.../cfg/` 同名文件）  
> 代码：`src/grid_map_memory_node.cpp`

---

## 1. 边缘形态学 / 投影误点问题

### 1.1 图中现象

可通行（绿点）与障碍物（垃圾桶等）交界处，绿色「地面」点爬上障碍物竖直表面；右侧 3D/栅格出现悬空或错误障碍簇。常见成因：

1. **分割在物体边界误标为 freespace**（语义泄漏）
2. **深度在深度不连续处混叠**（边界像素深度落在障碍物或背景）
3. **DepthToGrid 仅按 mask 判地面**，不做高度交叉校验

### 1.2 当前「边缘」相关逻辑（代码真实行为）

| 操作 | 参数 | 作用对象 | 是否覆盖「物体/地面交界」 |
|------|------|----------|---------------------------|
| **图像画框边缘强制 freespace** | `occupancy_grid.expand_pixels`（默认 5） | **整幅图像外框** 内侧 `N` 像素 | **否**。只处理相机画面四周，不是障碍物轮廓 |
| **指定类别膨胀** | `boundary_dilate_width` + `expand_classes` | 仅 `expand_classes` 中的类别 | **当前配置几乎否**。默认只膨胀 `["stair"]`，不膨胀障碍，也**不腐蚀** freespace |
| **高度地面矩形** | `ground_region_*` | 仅 **点云路径** `grid_generate_from_pointcloud` | DepthToGrid（mask+depth）**不走**此规则 |
| **点云密度滤波** | `pointcloud.filter.*` | ROI 内 **non_ground** 半径离群点 | 默认 `enabled: false`；且删的是障碍点疏离，**不修正「地面点上墙」** |

#### A. `expand_pixels`（图像外框强制可通行）

在 `grid_generate_point_cloud_and_grid` 中：

```text
若像素 (u,v) 落在图像四边 expand_pixels 带内
  → 强制 is_ground = true，label = 0（freespace）
```

用途：弱化 FOV 边缘畸变/深度失效带来的假障碍，**不是**物体轮廓处理。

#### B. `boundary_dilate_width` + `expand_classes`（类别膨胀，不是腐蚀）

在 `generateGrid` → `dilateLabels`：

```text
对 expand_classes 中每个类别做 cv::dilate
核半径 = boundary_dilate_width
膨胀结果写回 label_map（扩张该类区域）
```

当前 yaml：

```yaml
expand_pixels: 5
boundary_dilate_width: 5
expand_classes: ["stair"]   # 只扩楼梯，不处理垃圾桶/背景
```

**没有**对 freespace 做 `erode`，也**没有**对 `background(255)` / 障碍类做向内扩张。

#### C. DepthToGrid 的地面判定（与垃圾桶绿点直接相关）

```text
非图像外框像素：
  label ∈ freespace_classes（默认 freespace、stair）→ 地面
  其它已知类 / 未知类 → 非地面
  （不看 body z 高度）
```

因此：**mask 把垃圾桶底部标成 freespace，深度投影到竖直面上也会画成绿点并进地面点云。**

### 1.3 为何没 cover 图中场景

1. 现有形态学只服务「画面外框可通行」和「楼梯区域扩大」，**不针对障碍物-地面交界**。
2. mask+depth 路径**无**「语义 freespace ∩ 高度合理」双重门控。
3. 点云层半径滤波默认关闭，且目标是 ROI 内疏障碍点，不是边界错投。

### 1.4 可行改进方向（未实现，供选型）

**图像层（推荐先试，参数化）**

- 对 `background` / 非 freespace 做轻微 **dilate**，或对 freespace 做 **erode**，吃掉边界 1–3 像素
- 深度不连续处剔除：邻域深度梯度 `|d - d_neighbor| > thr` 的像素不投影
- 配置示例方向：`obstacle_dilate_width` / `freespace_erode_width` / `depth_edge_grad_m`

**点云层**

- 语义为 ground，但 body `z` 超出合理地面带（如 `z > freespace_min_z` 且不在 stair）→ 改标障碍或丢弃
- 垂直面上的「地面」簇：同格子内若有更高障碍，地面不落栅或占用取障碍优先（当前栅格是先 ground 再 non_ground 覆盖，障碍可压过；图中更像 **错误 ground 点本身被标成绿**）

**分割上游**

- 修边界、加大障碍类训练；本节点无法完全替代

### 1.5 相关参数是否已配置化

| 参数路径 | 已是 yaml 参数 | 说明 |
|----------|----------------|------|
| `occupancy_grid.expand_pixels` | 是 | 图像外框强制 freespace |
| `occupancy_grid.boundary_dilate_width` | 是 | 膨胀核半径 |
| `occupancy_grid.expand_classes` | 是 | 当前仅 stair |
| `occupancy_grid.freespace_classes` | 是 | 哪些 mask 名算地面 |
| `occupancy_grid.freespace_*_z` / `ground_region_*` | 是 | **仅点云路径** 用 |
| `pointcloud.filter.*` | 是 | 默认关 |
| freespace erode / 障碍 dilate / 深度梯度剔点 | **否** | 需新增 |

---

## 2. 总览：模式与记忆

```text
mode（点云前端 / 谁触发）
  depth_only | lidar_only | fusion
        ↓
  generateGrid（栅格生成）
    mask+depth | mask+depth+lidar | pointcloud_only
        ↓
  grid_publish_results
    memory.enable=true → odom 查位姿 → 3D voxel 记忆合并 → /map/occupancy_grid
    memory.enable=false → 直接发单帧 OccupancyGrid
```

- **mode** 决定谁触发、是否融合 depth/lidar 点云。
- **generateGrid 路径** 取决于 mask 是否同步成功，以及 `enable_lidar_grid_fusion`。
- **memory** 与 mode 正交：三种 mode 在 `memory.enable=true` 且 odom 命中时，都会对 **non_ground** 做近域记忆。

---

## 3. depth_only 完整流程

### 3.1 流水线

```text
[订阅]
  depth ──► fusion_depthImageCallback
  mask  ──► mask_cache
  rgb   ──► rgb_cache（仅 debug 可视化）
  odom  ──► odom_cache（仅记忆）

[depth 每帧]
  1) depth → fusion_depthImageToPointCloudOptimized
       相机系反投影 → camera_to_lidar → lidar_to_body
       可选 pointcloud.filter
       发布 /fused_point_cloud
  2) processDepthForGrid(depth, lidar=null)
  3) generateGrid(header=depth, depth_msg, pointcloud=null)

[generateGrid]
  A. Sync（当前优先 Strategy0：最新 mask → 配 depth/rgb）
  B. 有 mask+对齐 depth：
       convert → dilateLabels(expand_classes) → DepthToGrid
         · 按 mask∈freespace_classes 分 ground / non_ground
         · 图像外框 expand_pixels 强制 ground
         · 投影 body 系 2D OccupancyGrid
       可选 enable_lidar_grid_fusion（depth_only 通常无 lidar 缓存）
  C. 无 mask：
       pointcloud_only（用已发布 fused 点云或空则跳过）
         · 按 z / ground_region 分地面障碍 → 栅格

[publish]
  发 OccupancyGrid；若 memory.enable：
    用 header.stamp 查 odom → insert(non_ground 近域) → prune → merge → 发布合并栅格
```

### 3.2 地面 / 非地面（mask+depth）

| 条件 | 结果 |
|------|------|
| 像素在图像 `expand_pixels` 边带 | **地面**（强制） |
| `class_mapping[label]` ∈ `freespace_classes` | **地面** |
| 其它（含 background=255） | **非地面（障碍）** |

栅格写入：先写地面 cell，再写非地面 cell（同格非地面可覆盖）。  
语义占用：`label==0 → 0`，`255 → 100`，其它 `label+49`。

### 3.3 关键参数（depth_only）

| 段 | 关键参数 | 含义 |
|----|----------|------|
| `mode` | `depth_only` | 深度触发 |
| `topics.depth/mask/odom/...` | 话题 | 见 yaml |
| `camera.*` / `extrinsic.*` | 内参外参 | 点云与 DepthToGrid 各用一套外参 |
| `pointcloud.*` | 深度采样、滤波 | 前端 fused 点云 |
| `occupancy_grid.*` | sync、DepthToGrid、膨胀、栅格尺寸 | 见第 1、6 节 |
| `memory.*` | 是否记忆 | 见第 6 节 |
| `octomap_viz.*` | 可选可视化 | 默认关，不参与 `/map/occupancy_grid` |

---

## 4. lidar_only 完整流程

### 4.1 流水线

```text
[订阅]
  lidar ──► fusion_lidarCallback（depth 回调直接 return）
  mask/rgb 仍可订，但 generateGrid 以「无 depth」进入
  odom ──► 记忆用

[lidar 每帧]
  1) lidar → lidar_to_body 变换
  2) 可选滤波，发布 /fused_point_cloud
  3) generateGrid(header=lidar, depth_msg=null, pointcloud=filtered)

[generateGrid]
  A. Sync：无触发 depth 时 Strategy1/2 弱化；若有最新 mask，Strategy0 仍可能
     从 cache 另取一帧 depth 走 mask+depth（与「纯雷达模式」语义可能混用，需注意）
  B. 常见无 mask / sync 失败 → pointcloud_only：
       grid_generate_from_pointcloud
         is_ground =
           (mini_z < z < min_z) OR (z > max_z) OR (xy ∈ ground_region)
         地面格写 0，非地面写 100

[publish + memory]
  同 depth_only（对 non_ground 记忆）
```

### 4.2 点云地面判定（高度几何，无语义）

```text
is_ground =
  (freespace_mini_z < z < freespace_min_z)   // 近地薄层
  OR (z > freespace_max_z)                     // 过高也当「非障碍占用」侧的地面逻辑
  OR (x,y ∈ ground_region_*)                   // 机体脚下强制地面
否则 non_ground → 障碍 100
```

### 4.3 关键参数

| 段 | 参数 | 说明 |
|----|------|------|
| `mode` | `lidar_only` | |
| `topics.lidar` | 雷达话题 | |
| `extrinsic.lidar_to_body` | 雷达到 body | |
| `occupancy_grid.freespace_*_z` / `ground_region_*` | 地面几何 | |
| `occupancy_grid.range_*` | body 裁剪盒 | |
| `memory.*` | 同上 | |

---

## 5. fusion 完整流程

### 5.1 流水线

```text
[订阅]
  depth → push fusion_depth_cache；触发 fusion_tryTimeSync
  lidar → push fusion_lidar_cache；触发 fusion_tryTimeSync

[fusion_tryTimeSync]
  取双方最新帧，|t_depth - t_lidar| < pointcloud.time_sync_threshold_s
    是 → processDepthForGrid(depth, lidar)
         · 深度点云 + 雷达点云 fuse → /fused_point_cloud
         · generateGrid(..., depth, filtered_fused_or_lidar)
    否 → 仅用 lidar 仍可走「雷达侧」处理（日志 NOT synced）

[generateGrid]
  有 mask → mask+depth（DepthToGrid）
  若 enable_lidar_grid_fusion=true 且有雷达点云：
    另算雷达 OccupancyGrid，与 depth 栅格按 cell 取 max 合并
  无 mask → pointcloud_only（融合/滤波点云）

[publish + memory]
  同前
```

### 5.2 与 `enable_lidar_grid_fusion` 的区别

| 概念 | 配置 | 含义 |
|------|------|------|
| `mode: fusion` | 顶层 mode | 深度+雷达时间同步后出点云并触发栅格 |
| `occupancy_grid.enable_lidar_grid_fusion` | 默认 false | **另**把雷达栅格叠进 mask+depth 栅格 |

`mode=fusion` 不等于栅格里已经叠了雷达；要叠需再开 `enable_lidar_grid_fusion`。

---

## 6. 记忆（三模式共用）

`memory.enable: true` 时（三种 mode 共用路径）：

```text
grid_publish_results
  → odom_cache.GetClosestWithinThreshold(frame_stamp, odom_time_tolerance_s)
  → memoryCheckPeriodicClear(clear_interval_s)
  → memoryInsertFromCloud(non_ground)：
        body 矩形 [−rear, front] × [−right, left] 内点
        → T_odom 到 map → voxel_resolution 写入 hash(value=100)
  → memoryPruneOutside(keep_radius_m)
  → memoryMergeToBody：map 体素逆变换到 body，投影 2D，与当前帧取 max
  → publish 合并后 OccupancyGrid
```

| 参数 | 含义 |
|------|------|
| `memory.enable` | 开关 |
| `voxel_resolution` | map 体素边长 |
| `body_*_m` | 每帧写入记忆的机体近域 |
| `keep_radius_m` | 水平剪枝半径 |
| `clear_interval_s` | 定时全清；0=不清 |
| `odom_time_tolerance_s` | 位姿与帧时间容差；失败则发 raw 单帧栅格并 WARNING |

**说明：** 记忆只吃 **non_ground**；若边界误把障碍标成 ground，则不会进记忆，但会把栅格该格写成可通行（或语义占用），与图中「绿点上墙」一致。

---

## 7. generateGrid 三种源路径对照

| Grid Mode | 条件 | 地面判定 | 典型触发 mode |
|-----------|------|----------|---------------|
| `mask+depth` | sync 得到 mask+depth | **语义** `freespace_classes` + 图像外框 | depth_only / fusion |
| `mask+depth+lidar_only` | 上者 + `enable_lidar_grid_fusion` + 有雷达 | 语义栅格 ∪ 雷达栅格 max | fusion（也可 depth_only 有缓存时） |
| `pointcloud_only` | 无 mask | **高度 + ground_region** | lidar_only；depth 丢 mask 时回退 |

---

## 8. occupancy_grid / pointcloud / memory 参数速查

### occupancy_grid

| 参数 | 默认（主 yaml） | 作用 |
|------|-----------------|------|
| `time_sync_threshold_s` | 0.10 | mask/depth/rgb 同步容差 |
| `depth_min_m` / `depth_max_m` | 1e-5 / 5.0 | DepthToGrid 深度有效范围（乘 scale 后） |
| `skip_pixels` | 6 | 图像降采样步长 |
| `range_*` | ±5 / z∈[-0.7,5] | body 点有效盒 |
| `freespace_classes` | freespace, stair | 语义地面类名 |
| `expand_pixels` | 5 | **图像外框**强制 freespace |
| `boundary_dilate_width` | 5 | 类别膨胀半径 |
| `expand_classes` | [stair] | 被膨胀的类；**不含障碍交界处理** |
| `freespace_min/max/mini_z` | -0.1 / 1.5 / -0.7 | 点云路径地面高度 |
| `ground_region_*` | 机体脚下矩形 | 点云路径强制地面 |
| `resolution` / `size` | 0.05 / 200 | 2D 栅格 |
| `enable_lidar_grid_fusion` | false | 雷达栅格叠入 |
| `publish_debug_vis` / `publish_split_clouds` | 调试开关 | 板端建议 false |
| `class_mapping` | 0..4, 255 | mask 像素→类名 |

### pointcloud

| 参数 | 作用 |
|------|------|
| `depth_sample_step` / `depth_min_m` / `depth_max_m` | 前端深度点云 |
| `time_sync_threshold_s` | fusion 下 depth↔lidar |
| `filter.*` | ROI 半径离群点；默认关 |

### memory

见第 6 节。

### octomap_viz

与记忆无关；`publish_binary/markers=false` 时回调可直接空返回。

---

## 9. 模式对照简表

| 项目 | depth_only | lidar_only | fusion |
|------|------------|------------|--------|
| 触发源 | depth | lidar | depth↔lidar 同步 |
| 前端点云 | 深度→body | 雷达→body | 深度+雷达 fuse |
| 栅格主路径 | mask+depth（语义） | pointcloud_only（几何） | mask+depth，可选叠雷达栅格 |
| 地面依据 | mask 类名 | z + ground_region | 同左两列组合 |
| 记忆 | 可选，同管道 | 同左 | 同左 |
| 物体边界误投 | **易出现**（语义+无 erode） | 取决于几何阈 | 取决于 mask 路径 |

---

## 10. 与问题图的对应结论

1. **当前没有「障碍-地面轮廓腐蚀/膨胀」**；只有图像外框强制可通行 + stair 膨胀。  
2. 参数均已暴露在 `occupancy_grid` / `pointcloud.filter`；**没有**专门抑制图中边界错投的 erode/梯度参数。  
3. 点云层滤波可开，但**默认关**，且设计目标不是「绿点上墙」。  
4. 要 cover 该场景：优先图像对 freespace **erode** 或对 background **dilate**，或 DepthToGrid 增加 **语义地面 ∩ 高度合理**；二者都需新增逻辑。

---

## 11. D20 / D30 统一说明（2026-07-16）

### 原差异摘要

| 项 | 原 D20 | 原 D30 |
|----|--------|--------|
| 默认 mode | lidar_only | depth_only |
| YAML | legacy 三段 | v2 分层 |
| 记忆 | OcTree `enable_accumulation`（默认关） | 3D voxel hash `memory.enable` |
| 处理调度 | depth/lidar **callback 内重算** | 同左 |
| Pop 最新帧 | **不在 GridMap**；在上游 `lidar_seg` DataCache | GridMap `PopNewest` 只弹一帧不丢积压 |

### 统一后

- **同一份** `grid_map_memory_node.cpp` / `grid_map_common_type.h`
- 参数：**v2** 布局；D20=`mode: lidar_only`，D30=`mode: depth_only`
- 调度：callback 只 Push（可跳过更旧 stamp）→ 主循环 `spin_some` + `processPendingInputs` → `PopLatestByStamp` 只处理最新
- 记忆：voxel hash（`memory.enable`）；D20 默认 false；OcTree 仅 `octomap_viz`
