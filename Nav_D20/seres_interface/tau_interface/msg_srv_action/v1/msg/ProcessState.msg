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

# 进程状态(枚举型)
uint8 PROCESS_STATUS_DEAD=0x00      # 进程死亡
uint8 PROCESS_STATUS_RUNING=0x01    # 进程运行

uint8 module_id                     # 模块ID[必须使用 MODULE_ID_* 常量]
uint8 status                        # 进程状态[必须使用 PROCESS_STATUS_* 常量]