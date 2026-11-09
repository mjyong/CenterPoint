std_msgs/Header header
string session_id  # 多个客户端可同时订阅, 通过 session_id 区分
string action      # 动作类型: "started" / "result" / "error"
string code        # 状态码: "0" 表示成功
string desc        # 描述, 如 "success" 或错误信息
int32  seg_id      # 段ID
string text        # 本段转写文本
bool   is_final    # 是否为最终结果