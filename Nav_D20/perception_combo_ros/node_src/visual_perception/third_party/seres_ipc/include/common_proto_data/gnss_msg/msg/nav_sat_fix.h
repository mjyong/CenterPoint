#ifndef D10_APPLICATION_GNSS_NAV_SAT_FIX_H_
#define D10_APPLICATION_GNSS_NAV_SAT_FIX_H_

#include "common_msg/msg/header.h"
#include "nav_sat_status.h"

#pragma pack(1)

namespace seres_ipc {
namespace gnss_msg {
namespace msg {
struct NavSatFix {
    // Message header
    common_msg::msg::Header header;
    // Navigation satellite status
    NavSatStatus status;
    // Latitude in degrees
    double latitude;
    // Longitude in degrees
    double longitude;
    // Altitude in meters
    double altitude;
    // Position covariance matrix in row-major order
    double position_covariance[9];
    // Position covariance type
    uint8_t position_covariance_type;

    // Define enumeration constants
    static constexpr uint8_t COVARIANCE_TYPE_UNKNOWN = 0;
    static constexpr uint8_t COVARIANCE_TYPE_APPROXIMATED = 1;
    static constexpr uint8_t COVARIANCE_TYPE_DIAGONAL_KNOWN = 2;
    static constexpr uint8_t COVARIANCE_TYPE_KNOWN = 3;
};
} // namespace msg
} // namespace gnss_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_GNSS_NAV_SAT_FIX_H_
