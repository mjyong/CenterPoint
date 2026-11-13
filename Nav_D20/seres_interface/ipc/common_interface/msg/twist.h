#ifndef R11_TWIST_H_
#define R11_TWIST_H_

#include "utils/seres_vector.h"
#include <stdint.h>

#pragma pack(4)
namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Twist {
       Seres::vector<double, 3> linear;
       Seres::vector<double, 3> angular;
    };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc
#pragma pack()
#endif