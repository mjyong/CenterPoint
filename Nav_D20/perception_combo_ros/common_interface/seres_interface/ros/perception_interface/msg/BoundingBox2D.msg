string id                    # 目标唯一ID
string class_id              # 类别标签（如 "person"）
float64 score                # 置信度

geometry_msgs/Pose2D center  # 中心点（x,y）+ 旋转角 theta（通常0）
float64 size_x               # 宽度（像素）
float64 size_y               # 高度（像素）