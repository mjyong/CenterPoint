#ifndef D10_APPLICATION_FACE_RECO_MODULE_AUTH_RESULT_H_
#define D10_APPLICATION_FACE_RECO_MODULE_AUTH_RESULT_H_

#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace face_reco_module_msg {
namespace msg {
struct AuthResult {
    // Message header
    common_msg::msg::Header header;
    // Authentication success flag
    bool auth_success;
};
} // namespace msg
} // namespace face_reco_module_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_FACE_RECO_MODULE_AUTH_RESULT_H_
