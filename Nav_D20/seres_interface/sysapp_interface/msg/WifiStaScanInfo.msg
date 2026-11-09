# WiFi信号强度(枚举型)
uint8 WIFI_STA_STRENGTH_LEVEL_0=0x00        # 信号强度等级0
uint8 WIFI_STA_STRENGTH_LEVEL_1=0x01        # 信号强度等级1
uint8 WIFI_STA_STRENGTH_LEVEL_2=0x02        # 信号强度等级2
uint8 WIFI_STA_STRENGTH_LEVEL_3=0x03        # 信号强度等级3
uint8 WIFI_STA_STRENGTH_LEVEL_4=0x04        # 信号强度等级4
uint8 WIFI_STA_STRENGTH_LEVEL_5=0x05        # 信号强度等级5
# WiFi-STA加密方式(枚举型)
uint8 WIFI_STA_ENCRYPTION_NONE=0x01         # 未加密
uint8 WIFI_STA_ENCRYPTION_WPA2=0x02         # WPA2
uint8 WIFI_STA_ENCRYPTION_WPA=0x03          # WPA
uint8 WIFI_STA_ENCRYPTION_WEP=0x04          # WEP
uint8 WIFI_STA_ENCRYPTION_WAPI_PSK=0x05     # WAPI PSK
uint8 WIFI_STA_ENCRYPTION_WAPI_CERT=0x06    # WAPI CERT

string name                             # WiFi热点名称
uint8 strength                          # WiFi信号强度[必须使用 WIFI_STA_STRENGTH_LEVEL_* 常量]
uint8 encryption                        # WiFi-STA加密方式[必须使用 WIFI_STA_ENCRYPTION_* 常量]
bool is_connected    	                # 是否为当前连接的网络
bool is_save                            # 是否为已保存的网络
