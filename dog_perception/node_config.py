"""Parameters of the online node (``tools/dog/ros2_node.py``), free of ROS so they can be tested.

``robot:=kygi`` selects what the kygi665 robot needs and what its training sequences were built
with (``tools/dog/rosbag_to_sequence.py --inspvax-topic``): its topics, INSPVAX localization
(heading only, ENU origin at the first good fix, moved from the INS point 0.54 m behind the lidar to
the body origin), the RTK IMU in deg/s and INS axes, the 180 deg lidar yaw, the standing base height and ground levelling (``auto_ground`` / ``auto_level``).
Without them the network input online sits ~0.3 m off and keeps the 1-2 deg tilt the training data
does not have. Parameters given explicitly still override the preset.
"""
import numpy as np

from .geometry import make_T, rot_zyx
from .inspvax import KYGI_LIDAR_RPY, default_lever_arm
from .preprocess import DetFrameConfig, PreprocessConfig

# lidar height above the ground of the standing kygi robot: median of the per-scene estimates of
# the 0921 recordings (0.71 m over 29 scenes) and of the moving sample bag (0.72 m)
KYGI_STANDING_BASE_HEIGHT = 0.72

NODE_DEFAULTS = dict(
    detector="pillar", config="", checkpoint="", device="", predictor_model="",
    lidar_topic="/lidar_points", time_field="timestamp", time_mode="absolute",
    lidar_time_offset=0.0,            # [s] added to the lidar times (see tools/dog/time_offset_check.py)
    imu_topic="/imu/data",
    imu_convention="ros",             # ros: rad/s in the body frame | kygi: the RTK IMU (deg/s, INS axes)
    localization="odom",              # odom: nav_msgs/Odometry (LIO) | inspvax: kygi665_msgs/INSPVAX
    odom_topic="/Odometry", odom_twist="none",
    inspvax_topic="/rte/rdap/rtk/inspvax",
    ins_lever_arm=[float("nan")] * 3,  # [m] INSPVAX position point in the body frame (moved to the body origin);
                                       # nan: follows ins_azimuth_offset_deg (``ins_lever_arm``)
    ins_azimuth_offset_deg=180.0,     # heading = offset - azimuth: 180 with the dual antennas swapped, 0 once fixed
    extrinsic_t=[0.0, 0.0, 0.0], extrinsic_rpy=[0.0, 0.0, 0.0],
    base_height=0.45, auto_ground=False, auto_level=False,
    num_sweeps=5, world_frame="camera_init", publish_cloud=False,
)

ROBOT_PRESETS = {
    "kygi": dict(
        lidar_topic="/robot/topic/rdap/top_point_cloud2",
        imu_topic="/rte/rdap/rtk/imus/data_raw", imu_convention="kygi",
        localization="inspvax",
        extrinsic_rpy=[float(a) for a in KYGI_LIDAR_RPY],
        base_height=KYGI_STANDING_BASE_HEIGHT, auto_ground=True, auto_level=True,
    ),
}


def node_parameters(robot=""):
    """Defaults of every node parameter for ``robot`` ('' = generic LIO setup)."""
    if robot and robot not in ROBOT_PRESETS:
        raise ValueError("unknown robot %r (have: %s)" % (robot, ", ".join(sorted(ROBOT_PRESETS))))
    return {**NODE_DEFAULTS, **ROBOT_PRESETS.get(robot, {})}


def ins_lever_arm(params):
    """The INS lever arm the node applies: ``ins_lever_arm`` if given, else the one that goes with
    the antenna wiring (``ins_azimuth_offset_deg``): fitted on 0921 with the antennas swapped (180),
    the calibrated IMU centre once they are fixed (0). So after the fix only the offset changes."""
    L = np.asarray(params["ins_lever_arm"], np.float64)
    return L if L.shape == (3,) and np.isfinite(L).all() else np.asarray(default_lever_arm(params["ins_azimuth_offset_deg"]))


def preprocess_config(params):
    """PreprocessConfig from the node parameters (a dict like ``node_parameters``)."""
    rpy, t = params["extrinsic_rpy"], params["extrinsic_t"]
    return PreprocessConfig(T_body_lidar=make_T(rot_zyx(rpy[2], rpy[1], rpy[0]), t),
                            num_sweeps=int(params["num_sweeps"]),
                            det_frame=DetFrameConfig(base_height=float(params["base_height"]),
                                                     auto_ground=bool(params["auto_ground"]),
                                                     auto_level=bool(params["auto_level"])))
