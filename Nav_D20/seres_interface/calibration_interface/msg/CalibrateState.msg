
uint8 CALIB_SUCCESS = 0                       # 成功
uint8 CALIB_FAILED = 1                        # 
uint8 CALIB_STA_CANCELED = 2                  # 被取消 

string uuid                         # 请求操作的id, 当请求成功，异步返回执行结果时会携带这个id返回来;
uint8 command                       # 请求时的命令ID：COMMAND_*
uint8 state

string message                      # 可选
sensor_msgs/CompressedImage image   # 标定图像，可选 