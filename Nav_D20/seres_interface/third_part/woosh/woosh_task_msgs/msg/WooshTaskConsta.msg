# 可跳过的任务执行失败或取消后还能继续执行后续任务
# 不可跳过的任务执行失败或取消后将不能继续后续任务(等待重做)
#
# 不变的自定义字段

# 是否可跳过
bool can_skip
# 任务失败是否自动重做
bool fail_redo
# 任务失败是否自动开始下一个任务
bool fail_next
# 任务取消是否自动开始下一个任务
bool cancel_next
# 任务取消结束任务集
bool cancel_end
# 任务失败结束任务集
bool fail_end
# 取消-跳过
bool cancel_skip
# 失败-跳过
bool fail_skip
