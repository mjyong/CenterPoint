#ifndef _APPLICATION_COMMON_HEADER_H_
#define _APPLICATION_COMMON_HEADER_H_

#include "time.h"
#include "utils/seres_string.h"
#include "utils/seres_vector.h"
#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
struct Header {
    // Current module name,根据物理架构模块名确定，命名不要超过32个字节
    Seres::string<32> module_name;
    // Timestamp in milliseconds
    uint64_t timestamp_ms;
    // Forward module names
    Seres::vector<Seres::string<32>, 128> pre_module_names;
    // Forward module timestamps
    Seres::vector<Time, 128> pre_module_recvtime;
    // Frame ID
    Seres::string<10> frame_id;
  };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_HEADER_H_
