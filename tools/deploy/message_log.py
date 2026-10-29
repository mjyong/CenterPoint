"""The online node's input messages as plain arrays (msgs.npz), for the C++ node path without ROS.

``bag_messages`` reads a rosbag2 in arrival order (the bag's receive times, as ``ros2 bag play`` replays
it); ``MessageLog`` keeps the fields ``dog_perception/online.py`` reads; ``dogperc_replay --messages``
(deploy/cpp, ``dogperc/message_log.hpp``) feeds them to the C++ OnlineInputs. Layout:

    order_kind, order_index        arrival order: kind 0 imu, 1 odom, 2 inspvax, 3 cloud; index among its kind
    imu_t, imu_w                   header stamp [s], angular_velocity (as published)
    odom_t, odom_p, odom_q, odom_v pose position, orientation (x, y, z, w), twist linear
    ins_t, ins_status, ins_pos_type, ins_lla (lat, lon, h), ins_azimuth, ins_vel (east, north, up)
    cloud_t, cloud_width, cloud_height, cloud_point_step, cloud_bigendian, cloud_layout,
    cloud_data_offsets, cloud_data  the PointCloud2 bytes, concatenated
    layout_offsets -> field_offset, field_datatype, field_count, field_name_offsets -> field_names
"""
from pathlib import Path

import numpy as np

KINDS = ("imu", "odom", "inspvax", "cloud")


def node_topics(params):
    """topic -> kind for the node parameters (``node_config.node_parameters``)."""
    pose = ("inspvax_topic", "inspvax") if params["localization"] == "inspvax" else ("odom_topic", "odom")
    return {params["imu_topic"]: "imu", params[pose[0]]: pose[1], params["lidar_topic"]: "cloud"}


def bag_messages(bag, params):
    """(kind, message) in the bag's order, the topics the node subscribes to."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.inspvax import register_kygi

    topics = node_topics(params)
    ts = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    with AnyReader([Path(bag)], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in topics]
        missing = set(topics) - {c.topic for c in conns}
        if missing:
            raise SystemExit("%s: no messages on %s" % (bag, ", ".join(sorted(missing))))
        for c, _, raw in r.messages(connections=conns):
            yield topics[c.topic], r.deserialize(raw, c.msgtype)


def _t(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class MessageLog:
    def __init__(self):
        self.order = []
        self.cols = {k: [] for k in ("imu_t", "imu_w", "odom_t", "odom_p", "odom_q", "odom_v", "ins_t", "ins_status",
                                     "ins_pos_type", "ins_lla", "ins_azimuth", "ins_vel", "cloud_t", "cloud_width",
                                     "cloud_height", "cloud_point_step", "cloud_bigendian", "cloud_layout")}
        self.counts = dict.fromkeys(KINDS, 0)
        self.data, self.layouts = [], {}

    def add(self, kind, msg):
        c = self.cols
        if kind == "imu":
            w = msg.angular_velocity
            c["imu_t"].append(_t(msg.header))
            c["imu_w"].append((w.x, w.y, w.z))
        elif kind == "odom":
            p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
            c["odom_t"].append(_t(msg.header))
            c["odom_p"].append((p.x, p.y, p.z))
            c["odom_q"].append((q.x, q.y, q.z, q.w))
            c["odom_v"].append((v.x, v.y, v.z))
        elif kind == "inspvax":
            c["ins_t"].append(_t(msg.header))
            c["ins_status"].append(int(msg.ins_status.data))
            c["ins_pos_type"].append(int(msg.pos_type.data))
            c["ins_lla"].append((msg.latitude, msg.longitude, msg.height))
            c["ins_azimuth"].append(msg.azimuth)
            c["ins_vel"].append((msg.east_velocity, msg.north_velocity, msg.up_velocity))
        else:
            layout = tuple((f.name, int(f.offset), int(f.datatype), int(f.count)) for f in msg.fields)
            c["cloud_t"].append(_t(msg.header))
            c["cloud_width"].append(int(msg.width))
            c["cloud_height"].append(int(msg.height))
            c["cloud_point_step"].append(int(msg.point_step))
            c["cloud_bigendian"].append(int(bool(msg.is_bigendian)))
            c["cloud_layout"].append(self.layouts.setdefault(layout, len(self.layouts)))
            self.data.append(np.frombuffer(bytes(msg.data), np.uint8))
        self.order.append((KINDS.index(kind), self.counts[kind]))
        self.counts[kind] += 1

    def save(self, path):
        shapes = dict(imu_w=3, odom_p=3, odom_q=4, odom_v=3, ins_lla=3, ins_vel=3)
        ints = {"ins_status", "ins_pos_type", "cloud_width", "cloud_height", "cloud_point_step", "cloud_bigendian",
                "cloud_layout"}
        out = {}
        for k, v in self.cols.items():
            a = np.asarray(v, np.int64 if k in ints else np.float64)
            out[k] = a.reshape(-1, shapes[k]) if k in shapes else a.reshape(-1)
        order = np.asarray(self.order, np.int64).reshape(-1, 2)
        out["order_kind"], out["order_index"] = order[:, 0], order[:, 1]
        sizes = [len(d) for d in self.data]
        out["cloud_data_offsets"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        out["cloud_data"] = np.concatenate(self.data) if self.data else np.zeros(0, np.uint8)
        fields = [f for layout in sorted(self.layouts, key=self.layouts.get) for f in layout]
        per = [len(layout) for layout in sorted(self.layouts, key=self.layouts.get)]
        out["layout_offsets"] = np.concatenate([[0], np.cumsum(per)]).astype(np.int64)
        names = [f[0].encode() for f in fields]
        out["field_name_offsets"] = np.concatenate([[0], np.cumsum([len(n) for n in names])]).astype(np.int64)
        out["field_names"] = np.frombuffer(b"".join(names), np.uint8).copy()
        out["field_offset"] = np.asarray([f[1] for f in fields], np.int64)
        out["field_datatype"] = np.asarray([f[2] for f in fields], np.int64)
        out["field_count"] = np.asarray([f[3] for f in fields], np.int64)
        np.savez_compressed(path, **out)
