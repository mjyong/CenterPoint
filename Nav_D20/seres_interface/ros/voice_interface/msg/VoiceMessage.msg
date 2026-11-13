# 异步语音消息，用于Topic通信
string ssid
string action
string method
string package
voice_interface/VoiceContent content

# ssid: 会话标识
#   UUID格式，用于标识一次完整的语音交互流程，由请求方生成
# action: 操作类型
#   TTS_STATE   - 播报状态回调（通用状态）
#   TTS_READY   - 准备播报
#   TTS_START   - 开始播报
#   TTS_PAUSE   - 暂停播报
#   TTS_RESUME  - 恢复播报
#   TTS_STOP    - 停止播报
#   TTS_COMPLETE - 完成播报
# method: 具体方法/补充信息
#   状态回调时可为空
#   停止播报时可指定原因: interrupt(被中断)、error(错误终止)
# package: 应用标识
#   用于标识消息来源，如"com.example.app"
# content: 数据内容
#   携带具体的语音内容数据，结构由VoiceContent.msg定义