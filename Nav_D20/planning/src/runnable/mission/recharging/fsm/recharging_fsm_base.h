/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 19:32:06
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */

#pragma once

#include <boost/sml.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <vector>
namespace sml = boost::sml;

namespace planning {
namespace mission {
class RechargingFSMBaseCtx;
// Recharging 充电执行子状态机（内层），仅在外层任务状态机
// (MissionFSMBase) 的 passive/standby/active 期间被驱动。
// 业务流程由任务触发标志区分：
//   进桩充电(back_stake=false): searching -> aligning -> inserting -> sitting
//                               -> done，失败经 standing_up/backing_off 重试
//   退桩(back_stake=true):      standing_up -> backing_off -> done
class RechargingFSMBase {
public:
  explicit RechargingFSMBase(RechargingFSMBaseCtx &ctx);
  // clang-format off
  // state
  enum class StateType {
    idle,        // 空闲，等待启动
    searching,   // 搜索/检测充电桩（外层 passive 期间）
    aligning,    // 对准充电桩，走向 ready position（外层 standby 期间）
    inserting,   // 进桩，走向 charging station（外层 active 期间）
    sitting,     // 坐下并等待充电判定（外层 active 期间）
    standing_up, // 起立（退桩流程起点 / 进桩失败重试）
    backing_off, // 退桩，退回 ready position
    done         // 完成
  };
  struct s_idle        {static constexpr auto name{"idle"};         static constexpr auto id{StateType::idle};       };
  struct s_searching   {static constexpr auto name{"searching"};    static constexpr auto id{StateType::searching};  };
  struct s_aligning    {static constexpr auto name{"aligning"};     static constexpr auto id{StateType::aligning};   };
  struct s_inserting   {static constexpr auto name{"inserting"};    static constexpr auto id{StateType::inserting};  };
  struct s_sitting     {static constexpr auto name{"sitting"};      static constexpr auto id{StateType::sitting};    };
  struct s_standing_up {static constexpr auto name{"standing_up"};  static constexpr auto id{StateType::standing_up};};
  struct s_backing_off {static constexpr auto name{"backing_off"};  static constexpr auto id{StateType::backing_off};};
  struct s_done        {static constexpr auto name{"done"};         static constexpr auto id{StateType::done};       };

  std::unordered_map<StateType, std::string> state_names = {
    {StateType::idle,         std::string(s_idle::name)},
    {StateType::searching,    std::string(s_searching::name)},
    {StateType::aligning,     std::string(s_aligning::name)},
    {StateType::inserting,    std::string(s_inserting::name)},
    {StateType::sitting,      std::string(s_sitting::name)},
    {StateType::standing_up,  std::string(s_standing_up::name)},
    {StateType::backing_off,  std::string(s_backing_off::name)},
    {StateType::done,         std::string(s_done::name)}
  };

  // event
  enum class EventType {
    start_charge,       // idle -> searching        （进桩充电流程启动）
    start_back_stake,   // idle -> standing_up      （退桩流程启动）
    detected,           // searching -> aligning    （检测到充电桩）
    arrived_ready,      // aligning -> inserting    （对准到位）
    arrived_station,    // inserting -> sitting     （进桩到位）
    retry,              // inserting -> standing_up （进桩超时重试）
    charge_ok,          // sitting -> done          （充电成功）
    charge_fail,        // sitting -> standing_up   （充电失败重试）
    motion_done,        // standing_up -> backing_off（起立完成）
    backoff_retry_done, // backing_off -> aligning  （退桩到位，重试进桩）
    backoff_finish,     // backing_off -> done      （退桩到位，流程完成）
    reset,              // any -> idle              （复位）
    spin
  };
  struct e_start_charge       {static constexpr auto name{"start_charge"};       static constexpr auto id{EventType::start_charge};      };
  struct e_start_back_stake   {static constexpr auto name{"start_back_stake"};   static constexpr auto id{EventType::start_back_stake};  };
  struct e_detected           {static constexpr auto name{"detected"};           static constexpr auto id{EventType::detected};          };
  struct e_arrived_ready      {static constexpr auto name{"arrived_ready"};      static constexpr auto id{EventType::arrived_ready};     };
  struct e_arrived_station    {static constexpr auto name{"arrived_station"};    static constexpr auto id{EventType::arrived_station};   };
  struct e_retry              {static constexpr auto name{"retry"};              static constexpr auto id{EventType::retry};             };
  struct e_charge_ok          {static constexpr auto name{"charge_ok"};          static constexpr auto id{EventType::charge_ok};         };
  struct e_charge_fail        {static constexpr auto name{"charge_fail"};        static constexpr auto id{EventType::charge_fail};       };
  struct e_motion_done        {static constexpr auto name{"motion_done"};        static constexpr auto id{EventType::motion_done};       };
  struct e_backoff_retry_done {static constexpr auto name{"backoff_retry_done"}; static constexpr auto id{EventType::backoff_retry_done};};
  struct e_backoff_finish     {static constexpr auto name{"backoff_finish"};     static constexpr auto id{EventType::backoff_finish};    };
  struct e_reset              {static constexpr auto name{"reset"};              static constexpr auto id{EventType::reset};             };
  struct e_spin               {static constexpr auto name{"spin"};               static constexpr auto id{EventType::spin};              };

  std::unordered_map<EventType, std::string> event_names = {
    {EventType::start_charge,       std::string(e_start_charge::name)},
    {EventType::start_back_stake,   std::string(e_start_back_stake::name)},
    {EventType::detected,           std::string(e_detected::name)},
    {EventType::arrived_ready,      std::string(e_arrived_ready::name)},
    {EventType::arrived_station,    std::string(e_arrived_station::name)},
    {EventType::retry,              std::string(e_retry::name)},
    {EventType::charge_ok,          std::string(e_charge_ok::name)},
    {EventType::charge_fail,        std::string(e_charge_fail::name)},
    {EventType::motion_done,        std::string(e_motion_done::name)},
    {EventType::backoff_retry_done, std::string(e_backoff_retry_done::name)},
    {EventType::backoff_finish,     std::string(e_backoff_finish::name)},
    {EventType::reset,              std::string(e_reset::name)},
    {EventType::spin,               std::string(e_spin::name)}
  };
  // clang-format on

  auto operator()() {
    using namespace boost::sml;
    /**
     * Initial state: *initial_state
     * Transition DSL: src_state + event [ guard ] / action = dst_state
     */
    // clang-format off
    return make_transition_table(
      * state<s_idle>        + event<e_start_charge>     / [this](){transition_action(EventType::start_charge);}      = state<s_searching>
      , state<s_idle>        + event<e_start_back_stake> / [this](){transition_action(EventType::start_back_stake);}  = state<s_standing_up>
      , state<s_idle>        + event<e_spin>             / [this](){spin_action(StateType::idle);}
      , state<s_idle>        + sml::on_entry<_>          / [this](){on_entry(StateType::idle);}
      , state<s_idle>        + sml::on_exit<_>           / [this](){on_exit(StateType::idle);}

      , state<s_searching>   + event<e_detected>         / [this](){transition_action(EventType::detected);}          = state<s_aligning>
      , state<s_searching>   + event<e_spin>             / [this](){spin_action(StateType::searching);}
      , state<s_searching>   + sml::on_entry<_>          / [this](){on_entry(StateType::searching);}
      , state<s_searching>   + sml::on_exit<_>           / [this](){on_exit(StateType::searching);}

      , state<s_aligning>    + event<e_arrived_ready>    / [this](){transition_action(EventType::arrived_ready);}     = state<s_inserting>
      , state<s_aligning>    + event<e_spin>             / [this](){spin_action(StateType::aligning);}
      , state<s_aligning>    + sml::on_entry<_>          / [this](){on_entry(StateType::aligning);}
      , state<s_aligning>    + sml::on_exit<_>           / [this](){on_exit(StateType::aligning);}

      , state<s_inserting>   + event<e_arrived_station>  / [this](){transition_action(EventType::arrived_station);}   = state<s_sitting>
      , state<s_inserting>   + event<e_retry>            / [this](){transition_action(EventType::retry);}             = state<s_standing_up>
      , state<s_inserting>   + event<e_spin>             / [this](){spin_action(StateType::inserting);}
      , state<s_inserting>   + sml::on_entry<_>          / [this](){on_entry(StateType::inserting);}
      , state<s_inserting>   + sml::on_exit<_>           / [this](){on_exit(StateType::inserting);}

      , state<s_sitting>     + event<e_charge_ok>        / [this](){transition_action(EventType::charge_ok);}         = state<s_done>
      , state<s_sitting>     + event<e_charge_fail>      / [this](){transition_action(EventType::charge_fail);}       = state<s_standing_up>
      , state<s_sitting>     + event<e_spin>             / [this](){spin_action(StateType::sitting);}
      , state<s_sitting>     + sml::on_entry<_>          / [this](){on_entry(StateType::sitting);}
      , state<s_sitting>     + sml::on_exit<_>           / [this](){on_exit(StateType::sitting);}

      , state<s_standing_up> + event<e_motion_done>      / [this](){transition_action(EventType::motion_done);}       = state<s_backing_off>
      , state<s_standing_up> + event<e_spin>             / [this](){spin_action(StateType::standing_up);}
      , state<s_standing_up> + sml::on_entry<_>          / [this](){on_entry(StateType::standing_up);}
      , state<s_standing_up> + sml::on_exit<_>           / [this](){on_exit(StateType::standing_up);}

      , state<s_backing_off> + event<e_backoff_retry_done> / [this](){transition_action(EventType::backoff_retry_done);} = state<s_aligning>
      , state<s_backing_off> + event<e_backoff_finish>     / [this](){transition_action(EventType::backoff_finish);}   = state<s_done>
      , state<s_backing_off> + event<e_spin>               / [this](){spin_action(StateType::backing_off);}
      , state<s_backing_off> + sml::on_entry<_>            / [this](){on_entry(StateType::backing_off);}
      , state<s_backing_off> + sml::on_exit<_>             / [this](){on_exit(StateType::backing_off);}

      , state<s_done>        + event<e_spin>             / [this](){spin_action(StateType::done);}
      , state<s_done>        + sml::on_entry<_>          / [this](){on_entry(StateType::done);}
      , state<s_done>        + sml::on_exit<_>           / [this](){on_exit(StateType::done);}

      // 任意状态 -> idle（复位），on_entry(idle) 清理执行状态
      , state<s_searching>   + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_aligning>    + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_inserting>   + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_sitting>     + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_standing_up> + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_backing_off> + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
      , state<s_done>        + event<e_reset>            / [this](){transition_action(EventType::reset);}             = state<s_idle>
    );
    // clang-format on
  }

private:
  void transition_action(EventType event);
  void spin_action(StateType state);
  void on_entry(StateType state);
  void on_exit(StateType state);

  RechargingFSMBaseCtx &ctx_;
};

} // namespace mission
} // namespace planning
