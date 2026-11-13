#ifndef D10_COMMON_INTERFACE_MSG_SIZE3_H
#define D10_COMMON_INTERFACE_MSG_SIZE3_H

#pragma pack(4)
namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Size3 {
        double length;
        double width;
        double height;
    };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc
#pragma pack()
#endif