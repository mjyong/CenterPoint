std_msgs/Header header       # 时间戳, 用于同步
string         session_id    # 会话标识(匹配 start)
uint16         sample_rate
uint16         channels
uint16         bits_per_sample
bool           is_big_endian
bool           is_signed
bool           is_end        # 标记最后一帧
uint8[]        data          # PCM 字节 (1280 或按格式计算)