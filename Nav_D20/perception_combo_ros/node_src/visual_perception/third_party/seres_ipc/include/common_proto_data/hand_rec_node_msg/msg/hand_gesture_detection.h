#ifndef D10_APPLICATION_HAND_REC_NODE_HAND_GESTURE_DETECTION_H_
#define D10_APPLICATION_HAND_REC_NODE_HAND_GESTURE_DETECTION_H_

#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace hand_rec_node_msg {
namespace msg {
struct HandGestureDetection {
    // Message header
    common_msg::msg::Header header;
    // Gesture class names
    char gesture_class[16][64];
    // Classification confidence
    float class_confidence[16];
};
} // namespace msg
} // namespace hand_rec_node_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_HAND_REC_NODE_HAND_GESTURE_DETECTION_H_
