int32 MSG_TYPE_FEEDBACK = 0     # 回调
int32 MSG_TYPE_COMMAND = 1      # 指令
int32 MSG_TYPE_RESULT = 2       # 结果

int32 type                      # msg请求类型
string token                    # 执行技能的token
string id                       # 执行技能的任务ID
string content                  # pb格式的数据