#适配层对外音频流接口
uint32 sample_rate    # 采样率（如44100）
uint8 channels        # 声道数（1或2）
uint8 format          # 格式
uint32 length         # 音频流长度
uint8[] data          # 音频的实际数据:PCM音频数据
int64 time_frame      # 时间戳（纳秒）