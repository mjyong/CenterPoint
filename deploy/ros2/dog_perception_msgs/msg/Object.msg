# One tracked object of the dog perception node (deploy/ros2/dogperc_ros), in the frame of ObjectArray.header:
# the det frame of the scan, gravity aligned, yaw following the body, origin on the ground under the robot.
uint8 LABEL_VEHICLE = 0
uint8 LABEL_PEDESTRIAN = 1
uint8 LABEL_CYCLIST = 2
uint8 MOTION_STATIC = 0            # not seen moving
uint8 MOTION_MOVING = 1
uint8 MOTION_STOPPED = 2           # moved before and stands now: may move again

uint32 id                          # track id, the same while the track lives
uint8 label
string label_name                  # vehicle / pedestrian / cyclist
uint8 motion
float32 score                      # track confidence
bool coasting                      # not detected in this scan, carried forward by the motion model
float32 age                        # [s] since the track started
geometry_msgs/Point position       # box centre [m]; z above the ground
float64 yaw                        # [rad] box heading (x along the length)
geometry_msgs/Vector3 size         # length, width, height [m]
geometry_msgs/Vector3 velocity     # over ground [m/s], z = 0
float64[16] covariance             # of [x, y, vx, vy], row major
float32[] prediction_times         # [s] after header.stamp
PredictedPath[] predictions        # most likely first; empty without a prediction
