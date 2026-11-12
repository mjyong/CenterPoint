#ifndef D10_1230_DETECTION_SEG_ARRAY_H_
#define D10_1230_DETECTION_SEG_ARRAY_H_

#include "common_interface/msg/header.h"
#include "utils/seres_vector.h"
#include "detection_object.h"
#include "segment_array.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct DetectionSegArray {
        common_interface::msg::Header header;
        Seres::vector<DetectionObject, 10> objects;
        Seres::vector<SegmentArray, 10> segs;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()
#endif
