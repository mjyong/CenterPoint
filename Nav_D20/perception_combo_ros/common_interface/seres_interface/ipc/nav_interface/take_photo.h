#ifndef D10_1230_TAKE_PHOTO_H_
#define D10_1230_TAKE_PHOTO_H_

#include "common_interface/msg/point.h"
#include "utils/seres_vector.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    
    struct TakePhoto {
        int sensor_id; // 0: visible light, 1: infrared
        int num_of_point;
        Seres::vector<common_interface::msg::Point, 100> target_point;
        int num_of_photo;
        double time_interval;
    };
}
}
}

#pragma pack()

#endif
