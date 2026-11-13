# topic  "/robot/topic/sysapp/factory_plc_data"
# 发布者：sysapp
# 订阅者：agent

uint64 timestamp             # 时间戳

int32 speed                  # 传送带速度（1、单位：mm/s 2、发送周期：100ms）
int32 station_id             # 工位ID
string car_color_code         # 颜色代码

string station_vin           # 工位VIN码 (1、发送周期：100ms)
string car_type_code         # 车型代码
int32 honger_speed           # 吊架速度
int32 honger_height          # 吊架高度
int32 honger_id              # 吊架ID

string station_vin_1         # 工位VIN码 (1、发送周期：100ms)
string car_type_code_1       # 车型代码
int32 honger_speed_1         # 吊架速度
int32 honger_height_1        # 吊架高度
int32 honger_id_1            # 吊架ID

int32 encoder_val_ir         # 红外触发点编码器值
int32 encoder_val            # 传送带实时编码器值
