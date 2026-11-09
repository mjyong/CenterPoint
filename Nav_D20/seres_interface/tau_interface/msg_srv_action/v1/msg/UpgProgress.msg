# 请求源(枚举型)
uint8 MODULE_ID_NONE=0x00                   # 未定义
uint8 MODULE_ID_PROCESS_MGR=0x01            # NadProcessMgr
uint8 MODULE_ID_MCU_COM=0x02                # NadMcuCom
uint8 MODULE_ID_LOG_SRV=0x03                # NadLogSrv
uint8 MODULE_ID_POWER_MGR=0x04              # NadPowerMgr
uint8 MODULE_ID_SYSTEM_MGR=0x05             # NadSystemMgr
uint8 MODULE_ID_TIME_SRV=0x06               # NadTimeSrv
uint8 MODULE_ID_UDS=0x07                    # NadUds
uint8 MODULE_ID_UPGRADE=0x08                # NadUpgrade
uint8 MODULE_ID_SYSTEM_TEST=0x09            # NadSystemTest
uint8 MODULE_ID_DATA_SRV=0x0A               # NadDataSrv
uint8 MODULE_ID_BT_SRV=0x0B                 # NadBtSrv
uint8 MODULE_ID_WIFI_SRV=0x0C               # NadWifiSrv
uint8 MODULE_ID_GNSS_SRV=0x0D               # NadGnssSrv
uint8 MODULE_ID_CCU_OTA=0x0E                # CcuOta

# 升级模式(枚举型)
uint8 UPG_MODE_IDLE=0x00                    # IDLE
uint8 UPG_MODE_DOIP=0x01                    # DOIP
uint8 UPG_MODE_USB=0x02                     # USB
uint8 UPG_MODE_FOTA=0x03                    # FOTA
uint8 UPG_MODE_FOTA_SELIENT=0x04            # FOTA_SELIENT

# 升级类型(枚举型)
uint8 UPG_TYPE_ALL=0x00                     # ALL
uint8 UPG_TYPE_ONLY_MCU=0x01                # ONLY_MCU
uint8 UPG_TYPE_ONLY_NAD=0x02                # ONLY_NAD

uint8 src                                   # 请求源[必须使用 MODULE_ID_* 常量]
uint8 mode                                  # 升级模式[必须使用 UPG_MODE_* 常量]
uint8 type                                  # 升级类型[必须使用 UPG_TYPE_* 常量]
int32 progress                              # 升级进度[0-开始执行升级，10-条件检查通过，50-NAD刷写成功，80-MCU刷写成功，90-设置备份分区为激活分区成功，99-复位成功，100-升级成功]
int32 status_code                           # 升级状态码[0-成功，1-升级中，2-未升级，3-升级失败]
string status_description                   # 升级状态描述