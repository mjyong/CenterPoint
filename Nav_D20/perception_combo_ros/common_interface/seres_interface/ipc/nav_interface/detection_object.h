#ifndef D10_1230_DETECTION_OBJECT_H_
#define D10_1230_DETECTION_OBJECT_H_

#include "utils/seres_vector.h"
#include "utils/seres_string.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct DetectionObject {
        Seres::string<64> class_name;
        uint8_t class_id;
        uint8_t state;
        float confidence;
        Seres::vector<float, 4> bounding_box;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
