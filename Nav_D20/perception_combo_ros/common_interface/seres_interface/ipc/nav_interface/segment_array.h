#ifndef D10_1230_SEGMENT_ARRAY_H_
#define D10_1230_SEGMENT_ARRAY_H_

#include "utils/seres_vector.h"
#include "utils/seres_string.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct SegmentArray {
        Seres::string<64> class_name;
        uint8_t class_id;
        float confidence;
        Seres::vector<float, 4> boundary;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()
#endif