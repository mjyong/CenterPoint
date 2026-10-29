# The tracked objects of one lidar scan, for the planner.
std_msgs/Header header             # stamp: the scan (sweep end, localization clock); frame_id: the det frame
                                   # (node parameter local_frame)
string world_frame                 # the localization frame (camera_init for LIO, the ENU origin for INSPVAX)
geometry_msgs/Pose world_pose      # pose of the header frame in world_frame at header.stamp
Object[] objects
