# UWB 遥控器按键事件
# 发布: bluetooth 模块（UWB 遥控器通过 BLE 上报）
# 订阅: 语音/急停/跟随/拍照等业务模块
# topic: /nad/topic/v1/UwbKeyEvent

# ============ 按键事件类型 ============
uint8 EVENT_PRESS_DOWN=0      # 按键按下
uint8 EVENT_PRESS_UP=1        # 按键抬起（仅语音按键关注，其他按键忽略此事件）

# ============ 按键 ID ============
uint8 BTN_VOICE=0             # 语音按键（关注 PRESS_DOWN / PRESS_UP）
uint8 BTN_EMERGENCY_STOP=1    # 急停按键（仅关注 PRESS_DOWN）
uint8 BTN_FOLLOW=2            # 跟随按键（按下即开始跟随）
uint8 BTN_CANCEL_FOLLOW=3     # 取消跟随按键（按下即取消跟随）
uint8 BTN_CAMERA=4            # 拍照/摄像按键（按下即触发拍照）

# ============ 消息字段 ============
std_msgs/Header header        # 标准头（含 timestamp / frame_id / seq）
uint8 event                   # 按键事件 [使用 EVENT_* 常量]
uint8 button_id                # 按键 ID [使用 BTN_* 常量]
