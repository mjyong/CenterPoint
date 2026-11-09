# 模块ID(枚举型)
uint8 MODULE_ID_NONE=0x00           # 未定义
uint8 MODULE_ID_PROCESS_MGR=0x01    # NadProcessMgr
uint8 MODULE_ID_MCU_COM=0x02        # NadMcuCom
uint8 MODULE_ID_LOG_SRV=0x03        # NadLogSrv
uint8 MODULE_ID_POWER_MGR=0x04      # NadPowerMgr
uint8 MODULE_ID_SYSTEM_MGR=0x05     # NadSystemMgr
uint8 MODULE_ID_TIME_SRV=0x06       # NadTimeSrv
uint8 MODULE_ID_UDS=0x07            # NadUds
uint8 MODULE_ID_UPGRADE=0x08        # NadUpgrade
uint8 MODULE_ID_SYSTEM_TEST=0x09    # NadSystemTest
uint8 MODULE_ID_DATA_SRV=0x0A       # NadDataSrv
uint8 MODULE_ID_BT_SRV=0x0B         # NadBtSrv
uint8 MODULE_ID_WIFI_SRV=0x0C       # NadWifiSrv
uint8 MODULE_ID_GNSS_SRV=0x0D       # NadGnssSrv

# 网络制式(枚举型)
uint8 NW_RADIO_TECH_UNKNOWN=0x00    # 未定义
uint8 NW_RADIO_TECH_2G=0x02         # 2G
uint8 NW_RADIO_TECH_3G=0x03         # 3G
uint8 NW_RADIO_TECH_4G=0x04         # 4G
uint8 NW_RADIO_TECH_5G=0x05         # 5G

uint8 module_id                     # 模块ID[必须使用 MODULE_ID_* 常量]
string plmn                         # 网络PLMN
string sim_operator                 # SIM卡运营商
uint8 rat                           # 网络制式[必须使用 NW_RADIO_TECH_* 常量]
int32 band                          # 频段
int32 tac                           # 跟踪区
int32 lac                           # 路由区
int32 psc                           # 优选扫描信道
int32 cell_id                       # 小区ID
int32 rssi                          # 信号强度
int32 rscp                          # 信号码功率
int32 rsrp                          # 下行参考信号功率
int32 rsrq                          # 下行特定小区参考信号接收质量
int32 sinr                          # 强度与干扰、噪声的比值