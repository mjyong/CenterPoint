# Node 恢复播放通知，从暂停中恢复播放，或从音量降低恢复正常音量播放
int32 AUDIOFOCUS_GAIN = 1                          

# Node 完全丢失焦点，停止播放，下次播放需重新申请焦点
int32 AUDIOFOCUS_LOSS = -1

# Node 暂时丢失焦点，停止播放，等恢复播放通知AUDIOFOCUS_GAIN再播放                    
int32 AUDIOFOCUS_LOSS_TRANSIENT = -2

# Node 暂时丢失焦点，降低音量播放，等恢复播放通知AUDIOFOCUS_GAIN再恢复正常音量
int32 AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK = -3

int32 focus_change

# Node 可以通过申请时自己传的listener来收焦点回调通知
string listener