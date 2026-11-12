#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_icp.py — 使用 ICP 拼接目录/列表下的 pcd 地图（Python 版）

- 以第一个点云（目录内按文件名排序）为基准坐标系
- 后续点云经 open3d 多尺度 ICP（由粗到细）配准到已拼接的总图
- 总图始终由【原始点云】变换后直接拼接，不做任何降采样，保留全部字段
  （positions / intensity / timestamp 等）
- --icp-voxel 仅用于加速 ICP 求解（求解用降采样副本），不影响总图，0 = ICP 也用全分辨率

用法:
    python3 merge_icp.py <pcd目录 | pcd文件...> [-o 输出.pcd] [--stages 8,4,2,1,0.5]
                         [--iters 50] [--icp-voxel 0.2] [--init "x,y,z,yaw_deg;..."]

    --init 手动初值，分号分隔，依次对应第 2..N 个点云（ICP 初值差时使用）
           例: --init "10,5,0,90"  或两个点云时: --init "10,5,0,90;0,0,0,45"
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d


def collect_files(inputs, output):
    """展开输入：目录按文件名排序，显式文件按给定顺序；排除输出文件"""
    files, dir_files = [], []
    for p in inputs:
        if os.path.isdir(p):
            dir_files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                          if f.lower().endswith('.pcd')]
        elif os.path.exists(p):
            files.append(p)
        else:
            sys.exit(f'路径不存在: {p}')
    # 目录排前面（有目录时以目录内容为主），否则保持用户给定顺序
    result = (dir_files + files) if dir_files else files
    result = [f for f in result if os.path.abspath(f) != os.path.abspath(output)]
    if len(result) < 2:
        sys.exit('至少需要两个 pcd 文件')
    return result


def to_legacy(tpcd):
    """tensor 点云 -> legacy 点云（float64, 仅 xyz），用于 ICP"""
    pc = o3d.geometry.PointCloud()
    pos = tpcd.point['positions'].numpy()
    pc.points = o3d.utility.Vector3dVector(pos.astype(np.float64))
    return pc


def make_init_T(x, y, z, yaw_deg):
    """x,y,z 平移 + 绕 z 轴 yaw 角（度）的 4x4 初值"""
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def icp_to_map(src, target_map, T_init, stages, iters, voxel):
    """多尺度 ICP: 将 src 配准到 target_map，返回 4x4 与最终评估"""
    src_icp = src.voxel_down_sample(voxel) if voxel > 0 else src
    map_icp = target_map.voxel_down_sample(voxel) if voxel > 0 else target_map
    T = T_init.copy()
    res = None
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    for d in stages:
        res = o3d.pipelines.registration.registration_icp(
            src_icp, map_icp, d, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        T = res.transformation.copy()
        print(f'    stage d={d:<5g} 收敛 fitness={res.fitness:.4f} rmse={res.inlier_rmse:.4f}')
    return T, res


def main():
    ap = argparse.ArgumentParser(description='ICP 拼接点云地图（总图用原始点云拼接，不降采样）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('用法:')[1] if '用法:' in __doc__ else None)
    ap.add_argument('inputs', nargs='+', help='pcd 目录或文件')
    ap.add_argument('-o', '--output', default='', help='输出 pcd（默认: 第一个输入目录/merged_icp.pcd）')
    ap.add_argument('--stages', default='8,4,2,1,0.5', help='ICP 多尺度对应距离，由粗到细')
    ap.add_argument('--iters', type=int, default=50, help='每个尺度的最大迭代次数')
    ap.add_argument('--icp-voxel', type=float, default=0.2,
                    help='ICP 求解用降采样体素(米)，仅加速求解不影响总图; 0=ICP全分辨率')
    ap.add_argument('--init', default='',
                    help='手动初值 "x,y,z,yaw_deg;..."，依次对应第2..N个点云')
    args = ap.parse_args()

    if not args.output:
        first = args.inputs[0]
        base = first if os.path.isdir(first) else os.path.dirname(first) or '.'
        args.output = os.path.join(base, 'merged_icp.pcd')

    stages = [float(s) for s in args.stages.split(',') if s.strip()]
    files = collect_files(args.inputs, args.output)
    print(f'输出: {args.output}')

    # ---- 读取原始点云（open3d tensor, 保留全部字段, 不降采样） ----
    tpcds = []
    for f in files:
        t = o3d.t.io.read_point_cloud(f)
        n = len(t.point['positions'])
        if n == 0:
            sys.exit(f'空点云: {f}')
        tpcds.append(t)
        attrs = ', '.join(k for k, _ in t.point.items() if k != 'positions')
        print(f'[{len(tpcds)-1}] {os.path.basename(f)}: {n} 点 (字段: positions, {attrs})')

    # 所有图的公共属性（positions 之外），缺失的图用 0 填充
    attr_names = sorted(set().union(*[set(dict(t.point.items()).keys()) for t in tpcds]) - {'positions'})

    # ---- ICP 逐帧配准并拼接（总图 = 原始点云直接变换拼接） ----
    map_legacy = to_legacy(tpcds[0])          # 已拼接总图（xyz, 用于 ICP target）
    merged_pos = tpcds[0].point['positions'].numpy()
    first_attrs = dict(tpcds[0].point.items())
    merged_attr = {k: first_attrs[k].numpy() if k in first_attrs
                   else np.zeros((len(merged_pos), 1), np.float32) for k in attr_names}
    poses = [np.eye(4)]

    # 手动初值
    inits = {}
    if args.init:
        parts = [p.strip() for p in args.init.split(';') if p.strip()]
        for i, p in enumerate(parts):
            x, y, z, yaw = [float(v) for v in p.split(',')]
            inits[i + 1] = make_init_T(x, y, z, yaw)

    for i in range(1, len(tpcds)):
        src_t = tpcds[i]
        name = os.path.basename(files[i])
        print(f'\n=== 配准 {name} ({len(src_t.point["positions"])} 点) -> 总图 ===')
        T_init = inits.get(i, np.eye(4))
        if i in inits:
            print(f'    使用手动初值: {inits[i][:3, 3]}')

        src_l = to_legacy(src_t)
        T, res = icp_to_map(src_l, map_legacy, T_init, stages, args.iters, args.icp_voxel)
        print(f'    最终 fitness={res.fitness:.4f} rmse={res.inlier_rmse:.4f}')
        print(f'    变换矩阵:\n{T}')
        if res.fitness < 0.05:
            print('    警告: 重合率过低, 配准可能失败! 可尝试 --init 提供初值或调 --stages')

        # 用最终变换拼接【原始点云】（全字段, 不降采样）
        pos = src_t.point['positions'].numpy()  # (n,3) float32
        pos_h = np.hstack([pos, np.ones((len(pos), 1), np.float32)]).astype(np.float64)
        pos = (pos_h @ T.T)[:, :3].astype(np.float32)

        # legacy 地图追加变换后的点（供下一轮 ICP target 用）
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(pos.astype(np.float64))
        map_legacy += tmp

        merged_pos = np.vstack([merged_pos, pos])
        src_attrs = dict(src_t.point.items())
        for k in attr_names:
            v = src_attrs[k].numpy() if k in src_attrs \
                else np.zeros((len(pos), 1), np.float32)
            merged_attr[k] = np.vstack([merged_attr[k], v])
        poses.append(T)
        print(f'    拼入后总图点数: {len(merged_pos)}')

    # ---- 写出总图（保留全部字段, 不降采样） ----
    out = o3d.t.geometry.PointCloud()
    out.point['positions'] = o3d.core.Tensor(merged_pos)
    for k in attr_names:
        out.point[k] = o3d.core.Tensor(merged_attr[k])
    if not o3d.t.io.write_point_cloud(args.output, out, compressed=True):
        sys.exit(f'保存失败: {args.output}')
    print(f'\n总图点数: {len(merged_pos)} (原始点数之和: '
          f'{sum(len(t.point["positions"]) for t in tpcds)}), 已保存: {args.output}')

    tf_path = args.output + '.transforms.txt'
    with open(tf_path, 'w') as f:
        for i, (fn, T) in enumerate(zip(files, poses)):
            f.write(f'# [{i}] {fn}\n')
            f.write('\n'.join('  '.join(f'{v:.15g}' for v in row) for row in T) + '\n\n')
    print(f'各帧变换矩阵已保存: {tf_path}')


if __name__ == '__main__':
    main()
