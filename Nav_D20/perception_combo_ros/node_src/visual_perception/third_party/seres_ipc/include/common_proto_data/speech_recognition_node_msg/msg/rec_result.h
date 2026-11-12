#ifndef D10_APPLICATION_SPEECH_RECOGNITION_NODE_REC_RESULT_H_
#define D10_APPLICATION_SPEECH_RECOGNITION_NODE_REC_RESULT_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace speech_recognition_node_msg {
namespace msg {
struct RecResult {
    // 定义枚举值常量
    static constexpr uint8_t CTRL_MOTION_STOP = 0;
    static constexpr uint8_t CTRL_MOTION_FORWARD = 1;
    static constexpr uint8_t CTRL_MOTION_BACKWARD = 2;
    static constexpr uint8_t CTRL_MOTION_T_LEFT = 3;
    static constexpr uint8_t CTRL_MOTION_T_RIGHT = 4;
    static constexpr uint8_t CTRL_MOTION_R_LEFT = 5;
    static constexpr uint8_t CTRL_MOTION_R_RIGHT = 6;
    static constexpr uint8_t CTRL_MOTION_COME_HERE = 7;
    static constexpr uint8_t CTRL_MOTION_HELLO = 8;
    static constexpr uint8_t CTRL_MOTION_NEW_YEAR_GREETINGS = 9;
    static constexpr uint8_t CTRL_MOTION_SIT_DOWN = 10;
    static constexpr uint8_t CTRL_MOTION_LIE_DOWN = 11;
    static constexpr uint8_t START_DEMONSTRATION = 12;
    static constexpr uint8_t WAIT_MOMENT = 13;
    static constexpr uint8_t CTRL_MOTION_CONTINUE = 14;
    static constexpr uint8_t QUIT = 15;
    static constexpr uint8_t CTRL_MOTION_PAUSE = 16;
    static constexpr uint8_t CTRL_MOTION_FACE_VERIFY_1 = 17;
};
} // namespace msg
} // namespace speech_recognition_node_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_SPEECH_RECOGNITION_NODE_REC_RESULT_H_
