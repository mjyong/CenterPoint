uint8 led_index    # 灯光序号 0：表示所有灯,1-n：表示相关状态灯序号
uint32 color_ctrl    #灯光控制
uint8 color_mode   # 调色模式 1：rgb 3色调节,2：固定红黄绿
uint16 frequency   #闪烁频率

# 状态灯控制
#当color_mode=1 - 使用RGB进行灯光颜色控制
#color_ctrl表示如下，
#将其拆解为4个独立的1字节从高到低：
#[High1][High2][Low2][Low1]
#[Low1]：BLUE - 表示蓝色编码号
#[Low2]：GREEN - 表示绿色编码号
#[High2]：RED - 表示红色编码号
#当color_mode=2 - 使用固定红黄绿颜色控制
#color_ctrl表示：
#0:灯灭，1：仅绿灯常亮, 2:仅黄灯常亮,3:仅黄灯闪烁(默认频率1s), 4:仅红灯常亮，5:仅红灯闪烁(默认频率500ms), 0xff:全亮
# frequency表示：
# 闪烁频率：0 - 常亮（调色模式为2时默认频率），1- n表示闪烁间隔，单位：1ms

# 照明灯控制
# color_ctrl：0 - 灯灭，1 - 灯亮
# led_index：0 - 所n有灯，1- n表示相关状态灯序号
# 其他参数不用设置，默认即可

# 氛围灯控制
#color_mode=1 - 使用RGB进行灯光颜色控制
#color_ctrl表示如下，
#将其拆解为4个独立的1字节从高到低：
#[High1][High2][Low2][Low1]
#[Low1]：BLUE - 表示蓝色编码号
#[Low2]：GREEN - 表示绿色编码号
#[High2]：RED - 表示红色编码号
# 其他参数不用设置，默认即可