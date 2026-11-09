int32 EVENT_FOCUS_PAUSE = 0 # 焦点暂停
int32 EVENT_FOCUS_RESUME= 1 # 焦点重新获得

# 焦点
int32 focus_event           # 焦点事件
int32 cur_focus             # 当前有效的焦点
int32 target_focus          # 暂停/重新获得的焦点
int32 reserved_focus        # 已申请成功的焦点
string token                # 模块的token