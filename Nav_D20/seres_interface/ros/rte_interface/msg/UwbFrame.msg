#全迹UWB原始数据
int32 sec
uint32 nanosec
uint32 sync_cnt
uint32 index
uint32 fob_id
uint16 fob_type
float32 distance   
float32 angle
float32 pitch
uint8 rssi_len
int8[6] rssi


#转换之后的数据，暂时弃用
#float32 r      #m
#float32 rad    #rad [-pi,pi];
#float32 x      #m
#float32 y      #m
#int32 state    #> = 0 normal,-1 wrong, -2 wrong distance, -3 wrong angle


