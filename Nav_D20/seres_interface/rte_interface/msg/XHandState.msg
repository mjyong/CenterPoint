float32 cur_position            # 实际位置
uint16  cur_torque              # 实际转矩(电流)

#uint16  temperature            # 高8bit掌板温度，低8bit关节板温度
uint8 palm_temperature       # 掌板温度（高 8bit）
uint8 joint_temperature      # 关节温度（低 8bit）

uint16  raw_position            # 原始位置

uint8   comm_temp               # 通信温度
uint8   joint_temp              # 关节温度

uint16  comm_board_err          # 通信板错误
uint16  joint_board_err         # 关节板错误
uint16  senser_board_err        # 传感器板错误

uint8   senser_id               # 传感器 ID

uint16    fx                      # 受力 X 方向（切向力：100*N）
uint16    fy                      # 受力 Y 方向
uint16    fz                      # 受力 Z 方向（法向力：100*N）
uint16    fd                      # 切向力的方向（角度）
uint32    proximity               # 接近觉
uint16    touch_status            # 触觉状态

uint8[360] forces               # 指尖传感器：360维（120点 × xyz）的力值
uint8[20] temp_array            # 指尖传感器：20个温度点的温度值
uint8  fingertip_temp           # 指尖传感器温度合值
