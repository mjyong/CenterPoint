uint32 id                    # 目标唯一跟踪ID
string class_name            # 类别标签（如 "person"）详见感知目标类型定义 
float32 score                # 置信度 [0..1]

geometry_msgs/Pose2D center  # 中心点（x,y）+ 旋转角 theta（通常0）
float32 size_x               # 宽度（像素）
float32 size_y               # 高度（像素）
