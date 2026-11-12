# DeviceLinkStates.msg
DeviceType device_type #设备类型
uint8 device_id	#若存在多个设备，该id表示相同设备不同链路
uint8 link_status      #链路状态
#1：通信正常
#0：通信断开
