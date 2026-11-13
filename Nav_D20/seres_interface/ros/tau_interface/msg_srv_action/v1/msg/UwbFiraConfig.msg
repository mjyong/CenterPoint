# UWB Fira 配置参数
# 用于蓝牙协商完成后将参数发布给底软（UWB 锚板），底软组装 UART 帧配置锚点启动测距

# TLV 类型(枚举型)
uint8 TLV_TYPE_CONFIG_UWB_FIRA_PARAM=0x24   # UWB Fira 参数配置帧

# PSDU 速率(枚举型)
uint8 PSDU_RATE_7_80_MBPS=0x00              # 7.80 Mbps (default)
uint8 PSDU_RATE_6_81_MBPS=0x01              # 6.81 Mbps
uint8 PSDU_RATE_27_2_MBPS=0x02              # 27.2 Mbps
uint8 PSDU_RATE_31_2_MBPS=0x03              # 31.2 Mbps
uint8 PSDU_RATE_850_KBPS=0x04               # 850 Kbps

# 测距模式(枚举型)
uint8 MULTI_MODE_UNICAST=0x00               # 单播(1对1测距)
uint8 MULTI_MODE_MULTICAST=0x01             # 多播(1对多测距)

std_msgs/Header header                      # 标准头

# --- TLV 头部 ---
uint8 type                                  # TLV 类型，固定 0x24 (TLV_TYPE_CONFIG_UWB_FIRA_PARAM)
uint8 len                                   # TLV 长度，固定 0x22 (34 字节)

# --- 版本与内层长度 ---
uint16 major_ver                            # 主版本，默认 0x0001
uint16 minor_ver                            # 次版本，默认 0x0000
uint8 inner_len                             # 内层 len，从 country_code 到 source_addr 长度，默认 0x17
uint8[2] country_code                       # 国家代码，默认 [0x4E, 0x43]

# --- UWB 会话参数（协商变化） ---
uint8[4] session_id                         # 会话 ID，蓝牙协商的非 0 随机值
uint8 preamble_idx                          # 前导码 BPRF:[9~12] HPRF:[25~32]，默认 0x0C
uint8 chan_num                              # 信道 5/6/7/8/9/10，默认 0x09
uint8 slot_ix                               # slot 数量 0x08=2 锚点 0x0C=4 锚点，默认 0x08
uint8 psdu_rate                             # PSDU 速率 [必须使用 PSDU_RATE_* 常量]，默认 0x00
uint8[2] slot_rstu                          # 时隙时间，默认 0x0960 (2400*0.8333us=2ms)
uint8[2] range_period                       # 测距周期，单位 ms，默认 0x0030 (48ms)
uint8 ranging_round_control                 # 测距轮次控制，默认 0x03
uint8[6] vupper48                           # 上层 48 位，默认 [01,01,01,01,01,01]，与 iPhone 交互会变化
uint8[2] destination_addr                   # UWB 钥匙短地址，与 iPhone 交互会变化
uint8[2] source_addr                        # 设备地址，随机产生发给手机，需配置到 UWB 锚点
uint8 num_of_anchors                        # 基站个数 0x02/0x04，若 slot_ix=0x0C 则为 4
uint8 multi_mode                            # 测距模式 [必须使用 MULTI_MODE_* 常量]，默认 0x01
uint8[2] vendor_id                          # 厂商 ID，Fira=0x0101 iPhone Nearby=0x004C
