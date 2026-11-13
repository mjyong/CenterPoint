std_msgs/Header header 
# 音频格式参数
uint8 channels          # 声道数 (1=单声道, 2=立体声)
uint32 sample_rate      # 采样率 (Hz)
uint8 bit_depth         # 位深度 (8, 16, 24, 32)
bool is_signed          # 是否为有符号数据
bool is_big_endian      # 是否为大端序
# 音频数据
uint8[] data            # PCM原始数据
uint32 length           # PCM数据长度