int32 msg_type   # 消息类型
int32 msg_id     # 消息ID
uint16 topic_id  # MQTT主题ID
uint16 exec_code # 执行码
uint16 resv      # 预留
uint16 data_len  # MQTT数据长度
uint8[] data     # MQTT数据