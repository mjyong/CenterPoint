#ifndef D10_APPLICATION_SPEECHRECOGNITIONNODE_SOUNDRECOGNITION_H_
#define D10_APPLICATION_SPEECHRECOGNITIONNODE_SOUNDRECOGNITION_H_

#include <cstdint>
#include "common_msg/msg/header.h"
#include "rec_result.h"

#pragma pack(1)

namespace seres_ipc {
namespace speech_recognition_node_msg {
namespace msg {
struct SoundRecognition {
    // 消息头
    common_msg::msg::Header header;
    // 识别结果
    RecResult result;
};
} // namespace msg
} // namespace speech_recognition_node_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_SPEECHRECOGNITIONNODE_SOUNDRECOGNITION_H_
