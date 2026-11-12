#ifndef D10_APPLICATION_GNSS_NAV_SAT_STATUS_H_
#define D10_APPLICATION_GNSS_NAV_SAT_STATUS_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace gnss_msg {
namespace msg {
struct NavSatStatus {
    // Enumeration value
    int8_t status;

    // Define enumeration constants
    static constexpr int8_t STATUS_NO_FIX = -1; // unable to fix position
    static constexpr int8_t STATUS_FIX = 0;     // unaugmented fix
    static constexpr int8_t STATUS_SBAS_FIX = 1; // with satellite-based augmentation
    static constexpr int8_t STATUS_GBAS_FIX = 2; // with ground-based augmentation

    // Enumeration value
    uint16_t service;

    // Define enumeration constants
    static constexpr uint16_t SERVICE_GPS = 1;
    static constexpr uint16_t SERVICE_GLONASS = 2;
    static constexpr uint16_t SERVICE_COMPASS = 4;
    static constexpr uint16_t SERVICE_GALILEO = 8;
};
} // namespace msg
} // namespace gnss_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_GNSS_NAV_SAT_STATUS_H_
