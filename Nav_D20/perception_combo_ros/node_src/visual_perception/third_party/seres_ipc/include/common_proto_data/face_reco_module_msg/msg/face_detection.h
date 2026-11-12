#ifndef D10_APPLICATION_FACE_RECO_MODULE_FACE_DETECTION_H_
#define D10_APPLICATION_FACE_RECO_MODULE_FACE_DETECTION_H_

#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace face_reco_module_msg {
namespace msg {
struct FaceDetection {
    // Message header
    common_msg::msg::Header header;
    // Person ID ("unknown" when unknown)
    char id[8];
    // Confidence (0 - 1)
    double score;
    // Male probability (0 - 1)
    float male_probability;
    // Person name
    char name[64];
};
} // namespace msg
} // namespace face_reco_module_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_FACE_RECO_MODULE_FACE_DETECTION_H_
