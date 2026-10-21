import numpy as np
import pytest

from dog_perception.detection import OracleDetector, OracleNoise
from dog_perception.geometry import inv_T
from dog_perception.io import SequenceReader, odom_row, write_sequence
from dog_perception.pipeline import PerceptionPipeline
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig, det_frame_from_body
from dog_perception.sim import make_scenario


def test_sequence_roundtrip_and_pipeline(tmp_path):
    sim = make_scenario(seed=3, duration=2.0)
    scans, infos = zip(*[sim.scan(k) for k in range(8)])
    imu_t, imu_w = sim.imu(0.0, 0.85)
    odom = [odom_row(t, *sim.odometry(t)) for t in np.arange(0.0, 0.85, 0.1)]
    write_sequence(str(tmp_path), scans, np.column_stack([imu_t, imu_w]), odom,
                   dict(T_body_lidar=sim.T_bl, base_height=0.45))
    rd = SequenceReader(str(tmp_path))
    assert len(rd) == 8 and np.allclose(rd.T_body_lidar, sim.T_bl)

    pipe = PerceptionPipeline(PreprocessConfig(T_body_lidar=rd.T_body_lidar, det_frame=DetFrameConfig(base_height=0.45)),
                              detector=None)
    oracle = OracleDetector(OracleNoise(false_positive_rate=0.0), seed=0)
    outs, k = [], 0
    last_t = -1.0
    for ev in rd.events(odom_latency=0.03):
        if ev[0] == "imu":
            assert ev[1] >= last_t - 0.05
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            scan = ev[1]
            T_wd = det_frame_from_body(pipe.poses.pose_at(scan.stamp), 0.45)
            out = pipe.on_scan(scan, detections=oracle(infos[k]["gt"], inv_T(T_wd), scan.stamp))
            k += 1
            if out is not None:
                outs.append(out)
            last_t = scan.stamp
    assert len(outs) == 8
    assert outs[-1].frame.points.shape[1] == 5
    assert len(outs[-1].tracks) > 0 and len(outs[-1].predictions) > 0
    tracks, preds = pipe.to_local(outs[-1])
    assert len(tracks) == len(outs[-1].tracks) and preds[0].frame == "det"
    assert {"preprocess_ms", "track_ms", "predict_ms", "total_ms"} <= set(outs[-1].timings)


def test_rosbag_converter(tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    from dog_perception.io import SequenceReader as SR
    import subprocess
    import sys

    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Time, Header, V3 = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"], T["geometry_msgs/msg/Vector3"]
    PF, PC2, Imu, Odom = (T["sensor_msgs/msg/PointField"], T["sensor_msgs/msg/PointCloud2"],
                          T["sensor_msgs/msg/Imu"], T["nav_msgs/msg/Odometry"])
    Q = T["geometry_msgs/msg/Quaternion"]

    def tm(t):
        s = int(np.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)))

    sim = make_scenario(seed=0, duration=1.0)
    t0 = 1.7e9
    bag = tmp_path / "bag"
    with Writer(bag, version=8) as w:
        cl = w.add_connection("/points", PC2.__msgtype__, typestore=ts)
        ci = w.add_connection("/imu", Imu.__msgtype__, typestore=ts)
        co = w.add_connection("/odom", Odom.__msgtype__, typestore=ts)
        for t, g in zip(*sim.imu(0, 0.25)):
            m = Imu(header=Header(stamp=tm(t0 + t), frame_id="imu"), orientation=Q(x=0., y=0., z=0., w=1.),
                    orientation_covariance=np.zeros(9), angular_velocity=V3(x=g[0], y=g[1], z=g[2]),
                    angular_velocity_covariance=np.zeros(9), linear_acceleration=V3(x=0., y=0., z=9.8),
                    linear_acceleration_covariance=np.zeros(9))
            w.write(ci, int((t0 + t) * 1e9), ts.serialize_cdr(m, Imu.__msgtype__))
        for t in (0.0, 0.1, 0.2):
            R, p, v = sim.odometry(t)
            q = odom_row(t, R, p)[4:8]
            pose = T["geometry_msgs/msg/PoseWithCovariance"](
                pose=T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=p[0], y=p[1], z=p[2]),
                                                 orientation=Q(x=q[0], y=q[1], z=q[2], w=q[3])), covariance=np.zeros(36))
            tw = T["geometry_msgs/msg/TwistWithCovariance"](
                twist=T["geometry_msgs/msg/Twist"](linear=V3(x=0., y=0., z=0.), angular=V3(x=0., y=0., z=0.)),
                covariance=np.zeros(36))
            m = Odom(header=Header(stamp=tm(t0 + t), frame_id="odom"), child_frame_id="body", pose=pose, twist=tw)
            w.write(co, int((t0 + t) * 1e9), ts.serialize_cdr(m, Odom.__msgtype__))
        scan, _ = sim.scan(1)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2"), ("timestamp", "<f8")])
        a = np.zeros(len(scan.xyz), dt)
        a["x"], a["y"], a["z"] = scan.xyz.T
        a["intensity"], a["timestamp"] = scan.intensity, scan.point_times + t0
        fields = [PF(name=n, offset=o, datatype=d, count=1) for n, o, d in
                  (("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("ring", 16, 4), ("timestamp", 18, 8))]
        m = PC2(header=Header(stamp=tm(t0 + 0.1), frame_id="hesai"), height=1, width=len(a), fields=fields,
                is_bigendian=False, point_step=dt.itemsize, row_step=dt.itemsize * len(a),
                data=np.frombuffer(a.tobytes(), np.uint8), is_dense=True)
        w.write(cl, int((t0 + 0.2) * 1e9), ts.serialize_cdr(m, PC2.__msgtype__))
    out = tmp_path / "seq"
    subprocess.run([sys.executable, "tools/dog/rosbag_to_sequence.py", "--bag", str(bag), "--out", str(out),
                    "--lidar-topic", "/points", "--imu-topic", "/imu", "--odom-topic", "/odom"], check=True)
    rd = SR(str(out))
    assert len(rd) == 1 and len(rd.imu) > 50 and len(rd.odom) == 3
    ev = [e for e in rd.events() if e[0] == "scan"][0][1]
    assert np.allclose(ev.xyz, scan.xyz) and np.allclose(ev.point_times, scan.point_times + t0)


def test_rosbag_converter_livox_and_cameras(tmp_path):
    """--lidar mid360: nanosecond point times, Livox noise tags and empty returns dropped, the preset's
    mounting / self filter in meta.json; camera messages copied unchanged into cameras.mcap and from
    there into another MCAP."""
    pytest.importorskip("rosbags")
    from mcap.reader import make_reader
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    import subprocess
    import sys
    from dog_perception.foxglove import McapWriter
    from dog_perception.io import SequenceReader as SR
    from dog_perception.lidars import LIDARS, register_compressed_video

    ts = register_compressed_video(get_typestore(Stores.ROS2_HUMBLE))
    T = ts.types
    Time, Header, V3, Q = (T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"], T["geometry_msgs/msg/Vector3"],
                           T["geometry_msgs/msg/Quaternion"])
    PF, PC2, Imu, Odom = (T["sensor_msgs/msg/PointField"], T["sensor_msgs/msg/PointCloud2"],
                          T["sensor_msgs/msg/Imu"], T["nav_msgs/msg/Odometry"])
    Vid = T["foxglove_msgs/msg/CompressedVideo"]

    def tm(t):
        s = int(np.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)))

    t0 = 1.79e9
    bag = tmp_path / "bag"
    rng = np.random.default_rng(0)
    with Writer(bag, version=8) as w:
        cl = w.add_connection("/points", PC2.__msgtype__, typestore=ts)
        ci = w.add_connection("/imu", Imu.__msgtype__, typestore=ts)
        co = w.add_connection("/odom", Odom.__msgtype__, typestore=ts)
        cc = w.add_connection("/cam_front", Vid.__msgtype__, typestore=ts)
        for k in range(30):
            t = t0 + 0.01 * k
            m = Imu(header=Header(stamp=tm(t), frame_id="imu"), orientation=Q(x=0., y=0., z=0., w=1.),
                    orientation_covariance=np.zeros(9), angular_velocity=V3(x=0., y=0., z=0.),
                    angular_velocity_covariance=np.zeros(9), linear_acceleration=V3(x=0., y=0., z=9.8),
                    linear_acceleration_covariance=np.zeros(9))
            w.write(ci, int(t * 1e9), ts.serialize_cdr(m, Imu.__msgtype__))
        for t in (0.0, 0.1, 0.2):
            pose = T["geometry_msgs/msg/PoseWithCovariance"](
                pose=T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=0., y=0., z=0.),
                                                 orientation=Q(x=0., y=0., z=0., w=1.)), covariance=np.zeros(36))
            tw = T["geometry_msgs/msg/TwistWithCovariance"](
                twist=T["geometry_msgs/msg/Twist"](linear=V3(x=0., y=0., z=0.), angular=V3(x=0., y=0., z=0.)),
                covariance=np.zeros(36))
            m = Odom(header=Header(stamp=tm(t0 + t), frame_id="odom"), child_frame_id="body", pose=pose, twist=tw)
            w.write(co, int((t0 + t) * 1e9), ts.serialize_cdr(m, Odom.__msgtype__))
        frames = [bytes(rng.integers(0, 255, 50 + k, dtype=np.uint8)) for k in range(3)]
        for k, data in enumerate(frames):                      # one key frame before the scan, two during it
            m = Vid(timestamp=tm(t0 + 0.04 * k), frame_id="camera", data=np.frombuffer(data, np.uint8), format="h264")
            w.write(cc, int((t0 + 0.04 * k) * 1e9), ts.serialize_cdr(m, Vid.__msgtype__))
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("tag", "u1"), ("line", "u1"),
                       ("timestamp", "<f8")])
        a = np.zeros(400, dt)
        a["x"], a["y"], a["z"] = rng.uniform(-5, 5, (3, 400)).astype(np.float32)
        a["intensity"], a["line"] = 10.0, np.arange(400) % 4
        a["timestamp"] = (t0 + 0.05 + 0.1 * np.arange(400) / 400) * 1e9
        a["tag"][:20] = 0x01                                    # spatial noise
        a["tag"][20:30] = 0x10                                  # other bits: kept
        a["x"][30:50] = a["y"][30:50] = a["z"][30:50] = 0.0     # no return
        fields = [PF(name=n, offset=o, datatype=d, count=1) for n, o, d in
                  (("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("tag", 16, 2), ("line", 17, 2),
                   ("timestamp", 18, 8))]
        m = PC2(header=Header(stamp=tm(t0 + 0.05), frame_id="livox"), height=1, width=len(a), fields=fields,
                is_bigendian=False, point_step=dt.itemsize, row_step=dt.itemsize * len(a),
                data=np.frombuffer(a.tobytes(), np.uint8), is_dense=True)
        w.write(cl, int((t0 + 0.16) * 1e9), ts.serialize_cdr(m, PC2.__msgtype__))
    out = tmp_path / "seq"
    subprocess.run([sys.executable, "tools/dog/rosbag_to_sequence.py", "--bag", str(bag), "--out", str(out),
                    "--lidar", "mid360", "--lidar-topic", "/points", "--imu-topic", "/imu", "--odom-topic", "/odom"],
                   check=True, capture_output=True)
    rd = SR(str(out))
    scan = [e for e in rd.events() if e[0] == "scan"][0][1]
    keep = np.r_[20:30, 50:400]
    assert len(scan.xyz) == len(keep) and np.allclose(scan.xyz, np.c_[a["x"], a["y"], a["z"]][keep])
    assert np.allclose(scan.point_times, a["timestamp"][keep] * 1e-9)
    m = rd.meta
    assert m["lidar"] == "MID360" and m["extrinsic_provisional"] and m["cameras"] == ["/cam_front"]
    assert m["self_filter"]["min_range"] == 0.4 and rd.preprocess_config().self_filter.body_boxes == \
        tuple(tuple(b) for b in LIDARS["mid360"].self_filter["body_boxes"])
    with open(rd.cameras, "rb") as f:
        got = [(s.name, s.encoding, c.message_encoding, ts.deserialize_cdr(msg.data, Vid.__msgtype__))
               for s, c, msg in make_reader(f).iter_messages()]
    assert [g[3].data.tobytes() for g in got] == frames
    assert {g[:3] for g in got} == {("foxglove_msgs/msg/CompressedVideo", "ros2msg", "cdr")}
    w2 = McapWriter(str(tmp_path / "copy.mcap"))
    assert w2.copy_from(rd.cameras, t0 + 0.01, None) == 2                 # from the second message on
    w2.close()
