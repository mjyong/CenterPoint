#ID 代表是哪个 对应JSON里面的 RGB_front_1920x1080
string id
# 相机名称
string camera_name
# 更新频率
uint32 update_rate
# 图像宽度
uint32 image_width
# 图像高度
uint32 image_height
# 相机内参矩阵  uint32 camera_matrix_rows
uint32 camera_matrix_cols
float64[] camera_matrix_data
# 畸变模型
string distortion_model
# 畸变系数  uint32 distortion_coefficients_rows
uint32 distortion_coefficients_cols
float64[] distortion_coefficients_data
# 投影矩阵  uint32 projection_matrix_rows
uint32 projection_matrix_cols
float64[] projection_matrix_data
