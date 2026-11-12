uint8 operating_mode                    # 工作模式：0-保持不变 1-定时器触发模式 2-命令触发模式 3-IO触发模式（暂不支持）
uint8 timer_reading_label_mode          # 定时器读取标签模式：0-保持不变 1-盘点EPC 2-读取BANK-用户区数据
uint16 read_label_retention_time_ms     # 读取的标签保持时间，单位ms，默认200ms，范围50~1000 0-保持不变
uint16 auto_triggering_time_interval    # 定时器自动触发时间间隔，单位ms，默认500ms，范围50~2000 0-保持不变
uint32 access_password                  # 访问密码，默认0x0000
uint8 antenna_power                     # 天线功率，0~30dBm 0-保持不变
uint8 frequency_parameters              # 频率参数：0-保持不变(出厂默认美标) 1-FCC（美标） 2-ETSI（欧标） 3-CHN（国标） 4-自定义（暂不支持）
uint8 link_parameters                   # 链路参数：0-默认均衡模式 1-低速高可靠模式 2-均衡模式 3-高速回复模式 4-超高速模式