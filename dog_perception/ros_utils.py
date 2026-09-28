"""PointCloud2 / header helpers shared by the bag converter and the ROS 2 node.

Work on both ``rosbags`` and ``rclpy`` message objects; no ROS import needed.
"""
import numpy as np

from .preprocess.pipeline import LidarScan

PF_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def stamp_of(header):
    s = header.stamp
    return s.sec + s.nanosec * 1e-9


def cloud_to_array(msg):
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in PF_DTYPES:
            continue
        names.append(f.name)
        formats.append(("<" if not msg.is_bigendian else ">") + PF_DTYPES[f.datatype])
        offsets.append(f.offset)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    data = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    return data


def scan_from_msg(msg, time_field, time_mode, intensity_field="intensity"):
    arr = cloud_to_array(msg)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    inten = arr[intensity_field].astype(np.float32) if intensity_field in arr.dtype.names else np.zeros(len(arr), np.float32)
    header_t = stamp_of(msg.header)
    if time_field in arr.dtype.names:
        t = arr[time_field].astype(np.float64)
        if time_mode == "relative":
            t = header_t + t
        elif time_mode == "relative_ns":
            t = header_t + t * 1e-9
    else:
        t = None
    ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    return LidarScan(xyz=xyz[ok], intensity=inten[ok], point_times=None if t is None else t[ok],
                     stamp=None if t is not None else header_t)
