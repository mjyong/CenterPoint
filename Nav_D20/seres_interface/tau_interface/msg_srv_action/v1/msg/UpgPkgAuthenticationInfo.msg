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

# 认证类型(枚举型)
uint8 UPG_AUTHENTICATION_TYPE_HASH=0x00     # 哈希校验
uint8 UPG_AUTHENTICATION_TYPE_PKI=0x01      # PKI校验

uint8 src                                   # 请求源[必须使用 MODULE_ID_* 常量]
string file_path                            # 升级包路径
uint8 type                                  # 认证类型[必须使用 UPG_AUTHENTICATION_TYPE_* 常量]
string param                                # 认证参数
int32 ret                                   # 认证结果，0-成功，1-失败