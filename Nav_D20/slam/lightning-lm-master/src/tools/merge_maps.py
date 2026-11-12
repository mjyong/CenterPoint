#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_maps.py — 将多个地图通过 ICP 拼接成一张大图

用法:
    python3 merge_maps.py \
        --input data/test/test_1 data/test/test_2 data/test/test_3 \
        --output data/test/merged \
        --stages 8,4,2,1,0.5

流程:
    1. 对每个地图读取:
       a) 3d/global.pcd (用于 ICP 配准)
       b) session/keyframes/*.yaml + *.pcd (关键帧元数据)
    2. 顺序增量 ICP:
       map[0] 为基准, map[1] 对 map[0] 做 ICP,
       map[2] 对拼接后总图做 ICP, 依次类推
    3. 将配准后的关键帧点云 + 变换后的 pose_opt 按顺序保存到输出目录:
          output/session/keyframes/<global_id>.pcd  <-- 已变换
          output/session/keyframes/<global_id>.yaml <-- 新 pose_opt
       并生成 3d/global.pcd (拼接)、3d/index.txt、global_map.yaml 等
    4. 保存每次 ICP 的变换矩阵与评估指标到 output/icp_result.yaml
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
import open3d as o3d


# ---------------------------------------------------------------------------
# 自定义 PCD 二进制读写器: 完整保留 intensity / timestamp 等自定义字段
# ---------------------------------------------------------------------------

_PCD_TYPES = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


class PcdCloud:
    """带自定义字段的点云容器: fields (字段名列表) + data (结构化 numpy 数组)"""

    def __init__(self, fields: List[str], data: np.ndarray):
        self.fields = fields
        self.data = data  # numpy structured array, shape=(N,)

    def __len__(self):
        return len(self.data)

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.data["x"], self.data["y"], self.data["z"]], axis=1)

    def has(self, name: str) -> bool:
        return name in self.fields

    def to_legacy(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        return pcd

    def transform(self, T: np.ndarray):
        """只变换 x y z, 其余字段 (intensity/timestamp 等) 原样保留"""
        R = T[:3, :3]
        t = T[:3, 3]
        pts = self.xyz
        self.data["x"], self.data["y"], self.data["z"] = (
            pts @ R.T + t).T

    @staticmethod
    def concat(clouds: List["PcdCloud"]) -> "PcdCloud":
        """拼接多个 PcdCloud (字段不一致时取并集, 缺失字段补 0)"""
        all_fields = []
        for c in clouds:
            for f in c.fields:
                if f not in all_fields:
                    all_fields.append(f)
        # 保序: x y z 优先
        base = ["x", "y", "z"]
        ordered = [f for f in base if f in all_fields] + \
                  [f for f in all_fields if f not in base]
        parts = []
        for c in clouds:
            arr = c.data
            dt_map = {name: arr.dtype[name] for name in arr.dtype.names}
            missing = [f for f in ordered if f not in dt_map]
            if missing or list(arr.dtype.names) != ordered:
                filled = np.zeros(len(arr), dtype=[(f, dt_map.get(f, "f4"))
                                                   for f in ordered])
                for name in arr.dtype.names:
                    filled[name] = arr[name]
                parts.append(filled)
            else:
                parts.append(arr)
        merged = np.concatenate(parts) if parts else np.zeros(
            0, dtype=[(f, "f4") for f in ordered])
        return PcdCloud(ordered, merged)


def _read_pcd_compressed_tensor(path: Path) -> PcdCloud:
    """用 Open3D Tensor API 读取 binary_compressed PCD (保留所有字段)"""
    tpcd = o3d.t.io.read_point_cloud(str(path))
    n = int(tpcd.point["positions"].shape[0])
    # 收集字段: positions -> x/y/z, 其余单列字段原样
    dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    extra_names = []
    for key in tpcd.point:
        if key == "positions":
            continue
        t = tpcd.point[key]
        if t.ndim == 2 and t.shape[1] == 1:
            name = {  # open3d 标准字段名映射
                "intensity": "intensity", "timestamp": "timestamp",
            }.get(key, key)
            np_kind = {  # open3d dtype -> numpy
                o3d.core.Dtype.Float32: "f4", o3d.core.Dtype.Float64: "f8",
                o3d.core.Dtype.UInt8: "u1", o3d.core.Dtype.UInt16: "u2",
                o3d.core.Dtype.UInt32: "u4", o3d.core.Dtype.UInt64: "u8",
                o3d.core.Dtype.Int8: "i1", o3d.core.Dtype.Int16: "i2",
                o3d.core.Dtype.Int32: "i4", o3d.core.Dtype.Int64: "i8",
            }.get(t.dtype)
            if np_kind is None:
                print(f"  [WARN] 跳过不支持的字段: {key} ({t.dtype})")
                continue
            dtype_list.append((name, np_kind))
            extra_names.append(key)

    arr = np.zeros(n, dtype=np.dtype(dtype_list))
    pos = tpcd.point["positions"].numpy()  # (N,3)
    if pos.dtype != np.float32:
        pos = pos.astype(np.float32)
    arr["x"], arr["y"], arr["z"] = pos.T
    for name, key in zip([d[0] for d in dtype_list[3:]], extra_names):
        arr[name] = tpcd.point[key].numpy().ravel()
    return PcdCloud([d[0] for d in dtype_list], arr)


def read_pcd_binary(path: Path) -> PcdCloud:
    """读取二进制 PCD (支持 binary / binary_compressed), 保留所有字段"""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"PCD 头部异常: {path}")
            header_lines.append(line)
            if line.startswith(b"DATA"):
                break
        data_type = ""
        for raw in b"".join(header_lines).decode("ascii", "replace").splitlines():
            toks = raw.split()
            if toks and toks[0].upper() == "DATA":
                data_type = toks[1]
                break

    if data_type == "binary_compressed":
        return _read_pcd_compressed_tensor(path)
    if data_type != "binary":
        raise RuntimeError(f"不支持的 DATA 类型: {data_type} ({path})")

    header = b"".join(header_lines)
    data_start = len(header)

    fields, sizes, types_, counts = [], [], [], []
    n_points = 0
    for raw in header.decode("ascii").splitlines():
        toks = raw.split()
        if not toks:
            continue
        key, vals = toks[0].upper(), toks[1:]
        if key == "FIELDS":
            fields = vals
        elif key == "SIZE":
            sizes = [int(v) for v in vals]
        elif key == "TYPE":
            types_ = vals
        elif key == "COUNT":
            counts = [int(v) for v in vals]
        elif key == "POINTS":
            n_points = int(vals[0])

    dtype_list = []
    for name, ty, sz, cnt in zip(fields, types_, sizes, counts):
        if (ty, sz) not in _PCD_TYPES:
            raise RuntimeError(f"不支持的字段类型 TYPE={ty} SIZE={sz} ({path})")
        base_t = _PCD_TYPES[(ty, sz)]
        dtype_list.append((name, base_t) if cnt == 1 else (name, base_t, cnt))
    dt = np.dtype(dtype_list)

    with open(path, "rb") as f:
        f.seek(data_start)
        data = np.fromfile(f, dtype=dt, count=n_points)

    if len(data) != n_points:
        raise RuntimeError(f"PCD 点数不符: 期望 {n_points}, 实际 {len(data)} ({path})")
    return PcdCloud(list(fields), data)


def write_pcd_binary(path: Path, cloud: PcdCloud):
    """写出二进制 PCD, 保留所有字段"""
    arr = cloud.data
    names = list(arr.dtype.names)
    if any(arr.dtype[name].base.kind not in "fiu" for name in names):
        raise RuntimeError("PCD 字段类型异常")

    _size_of = {"f4": 4, "f8": 8, "u1": 1, "u2": 2, "u4": 4, "u8": 8,
                "i1": 1, "i2": 2, "i4": 4, "i8": 8}
    _kind_of = {"f": "F", "u": "U", "i": "I"}

    sizes = [_size_of[arr.dtype[name].base.str[1:]] for name in names]
    types_ = [_kind_of[arr.dtype[name].base.kind] for name in names]
    counts = [arr.dtype[name].shape[0] if arr.dtype[name].shape else 1
              for name in names]

    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(names)}",
        f"SIZE {' '.join(str(s) for s in sizes)}",
        f"TYPE {' '.join(types_)}",
        f"COUNT {' '.join(str(c) for c in counts)}",
        f"WIDTH {len(arr)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(arr)}",
        "DATA binary",
    ]
    header = ("\n".join(lines) + "\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        arr.tofile(f)


def load_global_pcd(map_dir: Path) -> Tuple[o3d.geometry.PointCloud, PcdCloud]:
    """读取 3d/global.pcd; 同时返回保留字段的 PcdCloud. 失败返回 (None, None)"""
    global_pcd_path = map_dir / "3d" / "global.pcd"
    if not global_pcd_path.exists():
        print(f"[WARN] {global_pcd_path} 不存在")
        return None, None
    try:
        cloud = read_pcd_binary(global_pcd_path)
        print(f"  [INFO] 读取 global.pcd: {len(cloud)} 点, "
              f"字段: {' '.join(cloud.fields)}")
        return cloud.to_legacy(), cloud
    except Exception as e:
        print(f"[WARN] 读取 global.pcd 失败: {e}")
        return None, None


def load_keyframes(map_dir: Path) -> List[dict]:
    """
    读取一个地图的所有关键帧元数据 (不含重建点云).

    返回: keyframe_list: [{'global_id','session_id','cloud_file',
           'pose_opt':[x,y,z, qx,qy,qz,qw], ...}, ...]
    """
    kf_dir = map_dir / "session" / "keyframes"
    if not kf_dir.exists():
        raise RuntimeError(f"缺少关键帧目录: {kf_dir}")

    yaml_files = sorted(kf_dir.glob("*.yaml"), key=lambda p: int(p.stem))
    keyframes = []

    for yf in yaml_files:
        with open(yf, "r") as f:
            meta = yaml.safe_load(f)

        pose = meta.get("pose_opt", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if len(pose) == 3:
            pose = pose + [0.0, 0.0, 0.0, 1.0]
        elif len(pose) == 7:
            pose = [float(v) for v in pose]
        else:
            print(f"[WARN] 无法解析 pose_opt: {pose}, 跳过 {yf}")
            continue

        cloud_file = meta.get("cloud_file", f"{yf.stem}.pcd")
        cloud_path = kf_dir / cloud_file
        if not cloud_path.exists():
            cloud_path = kf_dir / f"{yf.stem}.pcd"
        if not cloud_path.exists():
            print(f"[WARN] 关键帧点云不存在: {cloud_path}")
            continue

        kf_info = {
            "global_id": int(meta.get("global_id", yf.stem)),
            "session_id": int(meta.get("session_id", 0)),
            "session_local_id": meta.get("session_local_id", 0),
            "timestamp": meta.get("timestamp", 0.0),
            "cloud_file": cloud_file,
            "pose_opt": pose,
            "pose_lio": meta.get("pose_lio", [0, 0, 0, 0, 0, 0, 1]),
            "pose_base": meta.get("pose_base", [0, 0, 0, 0, 0, 0, 1]),
            "raw_floor_id": meta.get("raw_floor_id", 1000),
            "floor_id": meta.get("floor_id", 1),
            "area_type": meta.get("area_type", "floor"),
            "transition_id": meta.get("transition_id", -1),
            "use_for_grid": meta.get("use_for_grid", True),
            "is_session_anchor": meta.get("is_session_anchor", False),
            "anchor_pose": meta.get("anchor_pose", [0, 0, 0, 0, 0, 0, 1]),
            "anchor_confidence": meta.get("anchor_confidence", 0.95),
        }
        try:
            kf_info["raw_floor_id"] = int(kf_info["raw_floor_id"])
        except Exception:
            kf_info["raw_floor_id"] = 1000
        try:
            kf_info["floor_id"] = int(kf_info["floor_id"])
        except Exception:
            kf_info["floor_id"] = 1

        keyframes.append(kf_info)

    if not keyframes:
        raise RuntimeError(f"地图 {map_dir} 没有有效关键帧")

    print(f"  [INFO] {map_dir.name}: {len(keyframes)} 关键帧")
    return keyframes


def pose_to_matrix(pose: list) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] -> 4x4 变换矩阵"""
    x, y, z, qx, qy, qz, qw = pose
    T = np.eye(4)
    qnorm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if qnorm < 1e-12:
        return T
    qx, qy, qz, qw = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm

    T[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    T[0, 1] = 2 * (qx * qy - qz * qw)
    T[0, 2] = 2 * (qx * qz + qy * qw)
    T[1, 0] = 2 * (qx * qy + qz * qw)
    T[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    T[1, 2] = 2 * (qy * qz - qx * qw)
    T[2, 0] = 2 * (qx * qz - qy * qw)
    T[2, 1] = 2 * (qy * qz + qx * qw)
    T[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def matrix_to_pose(T: np.ndarray) -> list:
    """4x4 变换矩阵 -> [x, y, z, qx, qy, qz, qw]"""
    x = float(T[0, 3])
    y = float(T[1, 3])
    z = float(T[2, 3])
    m00, m11, m22 = T[0, 0], T[1, 1], T[2, 2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (T[2, 1] - T[1, 2]) * s
        qy = (T[0, 2] - T[2, 0]) * s
        qz = (T[1, 0] - T[0, 1]) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m00 - m11 - m22))
        qw = (T[2, 1] - T[1, 2]) / s
        qx = 0.25 * s
        qy = (T[0, 1] + T[1, 0]) / s
        qz = (T[0, 2] + T[2, 0]) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m11 - m00 - m22))
        qw = (T[0, 2] - T[2, 0]) / s
        qx = (T[0, 1] + T[1, 0]) / s
        qy = 0.25 * s
        qz = (T[1, 2] + T[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m22 - m00 - m11))
        qw = (T[1, 0] - T[0, 1]) / s
        qx = (T[0, 2] + T[2, 0]) / s
        qy = (T[1, 2] + T[2, 1]) / s
        qz = 0.25 * s
    qnorm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if qnorm > 1e-12:
        qx, qy, qz, qw = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm
    return [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]


def align_icp_multiscale(source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         stages: List[float] = None,
                         voxel_size: float = 0.15,
                         max_iter: int = 50) -> Tuple[np.ndarray, dict]:
    """
    多尺度 ICP 配准 (由粗到细).

    Args:
        source: 待配准点云
        target: 目标点云
        stages: 各尺度对应距离 (m), 默认 [8, 4, 2, 1, 0.5]
        voxel_size: 下采样体素大小 (m), 仅用于加速
        max_iter: 每阶段最大迭代次数

    Returns:
        (4x4 变换矩阵, 评估字典)
    """
    if stages is None:
        stages = [8.0, 4.0, 2.0, 1.0, 0.5]

    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)

    target_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))

    T = np.eye(4)
    result = None

    for s in stages:
        print(f"  [ICP stage] d={s:.3f}m ...", end=" ")
        result = o3d.pipelines.registration.registration_icp(
            source_ds, target_ds, s, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        T = result.transformation.copy()
        print(f"fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}")

    result_p2l = o3d.pipelines.registration.registration_icp(
        source_ds, target_ds, stages[-1] * 0.5, T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    T = result_p2l.transformation.copy()

    info = {
        "fitness": float(result_p2l.fitness),
        "inlier_rmse": float(result_p2l.inlier_rmse),
        "correspondence_set_size": len(result_p2l.correspondence_set),
    }
    print(f"  [ICP Final] fitness={info['fitness']:.4f}, "
          f"inlier_rmse={info['inlier_rmse']:.4f}, "
          f"correspondences={info['correspondence_set_size']}")
    return T, info


def transform_pose_opt(pose: list, T_rel: np.ndarray) -> list:
    """将 pose_opt 左乘一个相对变换矩阵 T_rel"""
    T_pose = pose_to_matrix(pose)
    T_new = T_rel @ T_pose
    return matrix_to_pose(T_new)


def save_output(output_dir: Path, fused_keyframes: List[dict],
                fused_cloud: PcdCloud,
                raw_floor_id: int = 1000):
    """
    按照地图格式保存 (点云字段全量保留):
        output/session/keyframes/<global_id>.pcd
        output/session/keyframes/<global_id>.yaml
        output/3d/global.pcd
        output/3d/index.txt
        output/global_map.yaml
        output/2d/<timestamp>/map.yaml
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = output_dir / "session" / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    d3_dir = output_dir / "3d"
    d3_dir.mkdir(parents=True, exist_ok=True)

    sorted_kfs = sorted(fused_keyframes, key=lambda kf: kf["_global_order"])
    floors_kfs: dict = {}

    for idx, kf in enumerate(sorted_kfs):
        new_id = idx
        kf["global_id"] = new_id
        src_pcd = kf["_source_pcd"]
        pcd_path = kf_dir / f"{new_id}.pcd"
        # 直接复制原始关键帧点云 (局部坐标系, 保留 intensity/timestamp 等所有字段)
        shutil.copy2(str(src_pcd), str(pcd_path))

        fid = kf.get("floor_id", 1)
        if fid not in floors_kfs:
            floors_kfs[fid] = {"min_z": float("inf"), "max_z": float("-inf"),
                               "keyframes": []}
        floors_kfs[fid]["keyframes"].append(new_id)
        z = kf["pose_opt"][2]
        floors_kfs[fid]["min_z"] = min(floors_kfs[fid]["min_z"], z)
        floors_kfs[fid]["max_z"] = max(floors_kfs[fid]["max_z"], z)

        anchor_pose = kf.get("anchor_pose", [0, 0, 0, 0, 0, 0, 1])
        if len(anchor_pose) == 3:
            anchor_pose = anchor_pose + [0, 0, 0, 1]
        anchor_pose = [float(v) for v in anchor_pose]

        yaml_meta = {
            "global_id": new_id,
            "session_id": kf["session_id"],
            "session_local_id": kf.get("session_local_id", new_id),
            "timestamp": kf.get("timestamp", 0.0),
            "cloud_file": f"{new_id}.pcd",
            "pose_opt": kf["pose_opt"],
            "pose_lio": kf.get("pose_lio", [0, 0, 0, 0, 0, 0, 1]),
            "pose_base": kf.get("pose_base", [0, 0, 0, 0, 0, 0, 1]),
            "raw_floor_id": raw_floor_id,
            "floor_id": fid,
            "area_type": kf.get("area_type", "floor"),
            "transition_id": kf.get("transition_id", -1),
            "use_for_grid": True,
            "is_session_anchor": kf.get("is_session_anchor", False),
            "anchor_pose": anchor_pose,
            "anchor_confidence": kf.get("anchor_confidence", 0.95),
        }
        for key in ("pose_lio", "pose_base"):
            val = yaml_meta[key]
            if len(val) == 3:
                val = val + [0, 0, 0, 1]
            yaml_meta[key] = [float(v) for v in val]

        with open(kf_dir / f"{new_id}.yaml", "w") as f:
            yaml.dump(yaml_meta, f, default_flow_style=False, sort_keys=False)

    global_pcd_path = d3_dir / "global.pcd"
    write_pcd_binary(global_pcd_path, fused_cloud)
    print(f"  [INFO] 输出 global.pcd: {len(fused_cloud)} 点, "
          f"字段: {' '.join(fused_cloud.fields)}")

    index_lines = []
    for kf in sorted_kfs:
        pose = kf["pose_opt"]
        idx = kf["global_id"]
        line = f"{idx} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} ./{kf_dir.relative_to(output_dir)}/{idx}.pcd"
        index_lines.append(line)
    with open(d3_dir / "index.txt", "w") as f:
        f.write("\n".join(index_lines) + "\n")

    pts = fused_cloud.xyz
    if len(pts) > 0:
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
    else:
        min_xyz = [0, 0, 0]
        max_xyz = [0, 0, 0]

    ts = "merged"
    d2_dir = output_dir / "2d" / ts
    d2_dir.mkdir(parents=True, exist_ok=True)
    map_yaml = {
        "image": "map.png",
        "resolution": 0.05,
        "origin": [float(min_xyz[0]), float(min_xyz[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "mode": "trinary",
        "width": int(max_xyz[0] - min_xyz[0] + 1),
        "height": int(max_xyz[1] - min_xyz[1] + 1),
    }
    with open(d2_dir / "map.yaml", "w") as f:
        yaml.dump(map_yaml, f, default_flow_style=False)

    floors_list = []
    for fid, info in sorted(floors_kfs.items()):
        info["representative_z"] = (info["min_z"] + info["max_z"]) / 2.0
        floors_list.append({
            "floor_id": fid,
            "raw_floor_id": raw_floor_id + fid - 1,
            "representative_z": info["representative_z"],
            "min_z": info["min_z"],
            "max_z": info["max_z"],
            "map_image_file": f"2d/{ts}/map.png",
            "map_yaml_file": f"2d/{ts}/map.yaml",
            "keyframes": info["keyframes"],
        })

    global_map = {
        "version": 2,
        "map_uuid": "merged_map",
        "frame_id": "map",
        "multi_floor": True,
        "initial_raw_floor_id": raw_floor_id,
        "normalization_offset": 999,
        "floors": floors_list,
    }
    with open(output_dir / "global_map.yaml", "w") as f:
        yaml.dump(global_map, f, default_flow_style=False, sort_keys=False)

    print(f"  [INFO] 输出目录: {output_dir}")
    print(f"  [INFO] 关键帧总数: {len(sorted_kfs)}")
    print(f"  [INFO] 楼层数: {len(floors_list)}")


def main():
    parser = argparse.ArgumentParser(description="多地图 ICP 拼接 (顺序增量合并)")
    parser.add_argument("--input", nargs="+", required=True,
                        help="地图目录(按拼接顺序, 第一个为基准)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--voxel", type=float, default=0.15,
                        help="ICP 下采样体素大小(m), 默认 0.15")
    parser.add_argument("--dist", type=float, default=0.3,
                        help="ICP 最大对应距离(仅单尺度模式使用, 默认 0.3)")
    parser.add_argument("--iter", type=int, default=50,
                        help="ICP 最大迭代次数, 默认 50")
    parser.add_argument("--stages", default="8,4,2,1,0.5",
                        help="多尺度 ICP 对应距离(逗号分隔), 默认 8,4,2,1,0.5")
    parser.add_argument("--raw_floor_id", type=int, default=1000,
                        help="merged 地图 raw_floor_id 起始值")
    parser.add_argument("--no-global-pcd", action="store_true",
                        help="不使用 3d/global.pcd, 从关键帧重建点云做 ICP")
    args = parser.parse_args()

    input_dirs = [Path(p).resolve() for p in args.input]
    output_dir = Path(args.output).resolve()
    stages = [float(s.strip()) for s in args.stages.split(",") if s.strip()]

    print("=" * 60)
    print("多地图 ICP 拼接 (顺序增量合并)")
    print(f"  输入地图({len(input_dirs)} 张):")
    for i, p in enumerate(input_dirs):
        print(f"    [{i}] {p}")
    print(f"  输出: {output_dir}")
    print(f"  ICP stages: {stages}")
    print("=" * 60)

    # --- Step 1: 读取所有地图 ---
    all_map_kfs: List[List[dict]] = []
    all_map_clouds: List[o3d.geometry.PointCloud] = []  # 仅 xyz, 用于 ICP
    all_map_full_clouds: List[PcdCloud] = []           # 带字段, 用于拼接输出
    all_map_names: List[str] = []

    def rebuild_from_keyframes(d: Path, kfs: List[dict]) -> PcdCloud:
        """从关键帧重建带字段的点云 (pose_opt 变换到世界坐标)"""
        parts = []
        for kf in kfs:
            src_pcd = d / "session" / "keyframes" / kf["cloud_file"]
            if not src_pcd.exists():
                continue
            try:
                kf_cloud = read_pcd_binary(src_pcd)
            except Exception:
                pcd = o3d.io.read_point_cloud(str(src_pcd))
                arr = np.zeros(len(pcd.points), dtype=[("x", "f4"), ("y", "f4"),
                                                       ("z", "f4")])
                pts = np.asarray(pcd.points)
                arr["x"], arr["y"], arr["z"] = pts.T
                kf_cloud = PcdCloud(["x", "y", "z"], arr.astype(np.float32))
            kf_cloud.transform(pose_to_matrix(kf["pose_opt"]))
            parts.append(kf_cloud)
        return PcdCloud.concat(parts) if parts else PcdCloud(
            ["x", "y", "z"], np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]))

    for i, d in enumerate(input_dirs):
        if not d.exists():
            print(f"[ERROR] 目录不存在: {d}")
            sys.exit(1)
        print(f"\n[Step 1] 读取地图 [{i}] {d.name} ...")

        kfs = load_keyframes(d)
        for idx, kf in enumerate(kfs):
            kf["_source_pcd"] = d / "session" / "keyframes" / kf["cloud_file"]
            kf["_global_order"] = i * 100000 + idx

        all_map_kfs.append(kfs)
        all_map_names.append(d.name)

        if args.no_global_pcd:
            print(f"  [INFO] 从关键帧重建点云做 ICP ...")
            full = rebuild_from_keyframes(d, kfs)
            print(f"  [INFO] 重建点云: {len(full)} 点")
        else:
            _, full = load_global_pcd(d)
            if full is None:
                print(f"  [INFO] 从关键帧重建点云做 ICP ...")
                full = rebuild_from_keyframes(d, kfs)
                print(f"  [INFO] 重建点云: {len(full)} 点")

        all_map_full_clouds.append(full)
        all_map_clouds.append(full.to_legacy())

    # --- Step 2: 顺序增量 ICP 配准 ---
    fused_kfs = list(all_map_kfs[0])
    fused_cloud = all_map_full_clouds[0]  # 带字段
    fused_icp_cloud = all_map_clouds[0]   # 仅 xyz
    icp_results = []

    for i in range(1, len(input_dirs)):
        print(f"\n[Step 2.{i}] 地图[{i}] {all_map_names[i]} 对拼接结果做 ICP ...")
        print(f"  source: {len(all_map_full_clouds[i])} 点")
        print(f"  target (已拼接): {len(fused_cloud)} 点")

        source_icp = all_map_clouds[i]

        T_rel, info = align_icp_multiscale(source_icp, fused_icp_cloud, stages,
                                           args.voxel, args.iter)

        print(f"  [ICP] 变换矩阵 T[{i}]:")
        print(np.array2string(T_rel, precision=6, suppress_small=True))

        icp_results.append({
            "source_map": all_map_names[i],
            "target_map": "fused",
            "fitness": info["fitness"],
            "inlier_rmse": info["inlier_rmse"],
            "correspondence_set_size": info["correspondence_set_size"],
            "transformation": T_rel.tolist(),
        })

        new_kfs = []
        for kf in all_map_kfs[i]:
            new_kf = dict(kf)
            new_kf["pose_opt"] = transform_pose_opt(kf["pose_opt"], T_rel)
            if "pose_lio" in new_kf and len(new_kf["pose_lio"]) >= 3:
                lio_pose = list(new_kf["pose_lio"])
                if len(lio_pose) == 3:
                    lio_pose = lio_pose + [0, 0, 0, 1]
                T_lio = T_rel @ pose_to_matrix(lio_pose)
                new_kf["pose_lio"] = matrix_to_pose(T_lio)
            new_kfs.append(new_kf)

        # 带字段点云应用 ICP 变换 (intensity/timestamp 保留)
        src_full = all_map_full_clouds[i]
        src_full.transform(T_rel)
        fused_cloud = PcdCloud.concat([fused_cloud, src_full])
        # ICP 用的 xyz 点云同步拼接
        fused_icp_cloud = fused_icp_cloud + source_icp
        fused_kfs = fused_kfs + new_kfs
        print(f"  [INFO] 拼接后总点数: {len(fused_cloud)}")

    # --- Step 3: 保存 ---
    print(f"\n[Step 3] 保存结果到 {output_dir} ...")
    save_output(output_dir, fused_kfs, fused_cloud, args.raw_floor_id)

    icp_result_path = output_dir / "icp_result.yaml"
    with open(icp_result_path, "w") as f:
        yaml.dump({"icp_results": icp_results}, f, default_flow_style=False,
                  sort_keys=False)
    print(f"  [INFO] ICP 结果: {icp_result_path}")

    for i, d in enumerate(input_dirs):
        src_d2 = d / "2d"
        if src_d2.exists():
            for ts_dir in src_d2.iterdir():
                if ts_dir.is_dir():
                    dst_d2 = output_dir / "2d"
                    dst_d2.mkdir(parents=True, exist_ok=True)
                    for f in ts_dir.iterdir():
                        dst = dst_d2 / f.name
                        if not dst.exists():
                            try:
                                os.link(str(f), str(dst))
                            except OSError:
                                shutil.copy2(str(f), str(dst))
                            print(f"  [INFO] 链接 2d 文件: {f.name}")

    print("\n[DONE] 地图拼接完成")


if __name__ == "__main__":
    main()
