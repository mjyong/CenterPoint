std_msgs/Header header           # ROS2 标准消息头
uint8 led_index    # 灯光序号 0-n：表示氛围灯序号
#colorCtrl表示如下，
#将其拆解为4个独立的1字节从高到低：
#[High1][High2][Low2][Low1]
#[Low1]：BLUE - 表示蓝色编码号
#[Low2]：GREEN - 表示绿色编码号
#[High2]：RED - 表示红色编码号
#[High1]: 保留
uint32 color_ctrl    #颜色控制