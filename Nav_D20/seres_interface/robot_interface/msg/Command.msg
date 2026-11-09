uint64 seq            # 消息序列号
int32  type           # 消息命令类型，即对应的消息ID(协议定义)
string data           # 通信Agent做内部分发或上行sendtype==0时, 表示具体业务数据, 类型为type
                      # sendtype==1时, data为可额外携带回复数据,数据类型由resp_data_type决定.约束:仅用于传输文本字符串
uint8[] datastream    # 含义同data, 建议使用datastream.约束: 二进制和文本字符串均可(如"to_remote"话题)
uint8 sendtype        # 上行发送类型 send: 0（主动上报类）;response: 1（响应类）
uint8 remote_type     # 消息请求端类型 0-表示默认(无明确云端或app等);1-表示app端;2-表示云端
int32 qos             # sendtype==0 时有效，支持自定义QoS

int32 errcode         # sendtype==1 时有效
string msg            # sendtype==1 时有效
int32 resp_data_type  # sendtype==1 时有效 回复数据类型,resp_data_type!=0时,data携带回复数据
