#ifndef D10_APPLICATION_PLANNING_PLANNER_MOTION_H_
#define D10_APPLICATION_PLANNING_PLANNER_MOTION_H_

#include <cstdint>
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace planning_msg {
namespace msg {
struct PlannerMotion {
    // Message header
    common_msg::msg::Header header;
    // Velocity in the x - axis
    float vx;
    // Velocity in the y - axis
    float vy;
    // Yaw rate
    float vyaw;
    // Sport ID (enumeration value)
    uint32_t sport_id;

    // Define enumeration constants
    static constexpr uint32_t ROBOT_SPORT_API_ID_DAMP = 1001;
    static constexpr uint32_t ROBOT_SPORT_API_ID_BALANCESTAND = 1002;
    static constexpr uint32_t ROBOT_SPORT_API_ID_STOPMOVE = 1003;
    static constexpr uint32_t ROBOT_SPORT_API_ID_STANDUP = 1004;
    static constexpr uint32_t ROBOT_SPORT_API_ID_STANDDOWN = 1005;
    static constexpr uint32_t ROBOT_SPORT_API_ID_RECOVERYSTAND = 1006;
    static constexpr uint32_t ROBOT_SPORT_API_ID_EULER = 1007;
    static constexpr uint32_t ROBOT_SPORT_API_ID_MOVE = 1008;
    static constexpr uint32_t ROBOT_SPORT_API_ID_SIT = 1009;
    static constexpr uint32_t ROBOT_SPORT_API_ID_RISESIT = 1010;
    static constexpr uint32_t ROBOT_SPORT_API_ID_SWITCHGAIT = 1011;
    static constexpr uint32_t ROBOT_SPORT_API_ID_TRIGGER = 1012;
    static constexpr uint32_t ROBOT_SPORT_API_ID_BODYHEIGHT = 1013;
    static constexpr uint32_t ROBOT_SPORT_API_ID_FOOTRAISEHEIGHT = 1014;
    static constexpr uint32_t ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015;
    static constexpr uint32_t ROBOT_SPORT_API_ID_HELLO = 1016;
    static constexpr uint32_t ROBOT_SPORT_API_ID_STRETCH = 1017;
    static constexpr uint32_t ROBOT_SPORT_API_ID_TRAJECTORYFOLLOW = 1018;
    static constexpr uint32_t ROBOT_SPORT_API_ID_CONTINUOUSGAIT = 1019;
    static constexpr uint32_t ROBOT_SPORT_API_ID_CONTENT = 1020;
    static constexpr uint32_t ROBOT_SPORT_API_ID_WALLOW = 1021;
    static constexpr uint32_t ROBOT_SPORT_API_ID_DANCE1 = 1022;
    static constexpr uint32_t ROBOT_SPORT_API_ID_DANCE2 = 1023;
    static constexpr uint32_t ROBOT_SPORT_API_ID_GETBODYHEIGHT = 1024;
    static constexpr uint32_t ROBOT_SPORT_API_ID_GETFOOTRAISEHEIGHT = 1025;
    static constexpr uint32_t ROBOT_SPORT_API_ID_GETSPEEDLEVEL = 1026;
    static constexpr uint32_t ROBOT_SPORT_API_ID_SWITCHJOYSTICK = 1027;
    static constexpr uint32_t ROBOT_SPORT_API_ID_POSE = 1028;
    static constexpr uint32_t ROBOT_SPORT_API_ID_SCRAPE = 1029;
    static constexpr uint32_t ROBOT_SPORT_API_ID_FRONTFLIP = 1030;
    static constexpr uint32_t ROBOT_SPORT_API_ID_FRONTJUMP = 1031;
    static constexpr uint32_t ROBOT_SPORT_API_ID_FRONTPOUNCE = 1032;
};
} // namespace msg
} // namespace planning_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_PLANNING_PLANNER_MOTION_H_
