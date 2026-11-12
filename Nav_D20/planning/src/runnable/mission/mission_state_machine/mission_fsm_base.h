/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 19:32:06
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-07 10:44:24
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
class MissionFSMBaseCtx;
class MissionFSMBase {
public:
  explicit MissionFSMBase(MissionFSMBaseCtx &ctx);
  // clang-format off
  // state
  enum class StateType { 
    off, 
    passive, 
    standby, 
    active, 
    standstill, 
    fault 
  };
  struct s_off        {static constexpr auto name{"off"};         static constexpr auto id{StateType::off};       };
  struct s_passive    {static constexpr auto name{"passive"};     static constexpr auto id{StateType::passive};   };
  struct s_standby    {static constexpr auto name{"standby"};     static constexpr auto id{StateType::standby};   };
  struct s_active     {static constexpr auto name{"active"};      static constexpr auto id{StateType::active};    };
  struct s_standstill {static constexpr auto name{"standstill"};  static constexpr auto id{StateType::standstill};};
  struct s_fault      {static constexpr auto name{"fault"};       static constexpr auto id{StateType::fault};     };

  std::unordered_map<StateType, std::string> state_names = {
    {StateType::off,         std::string(s_off::name)},
    {StateType::passive,     std::string(s_passive::name)},
    {StateType::standby,     std::string(s_standby::name)},
    {StateType::active,      std::string(s_active::name)},
    {StateType::standstill,  std::string(s_standstill::name)},
    {StateType::fault,       std::string(s_fault::name)}
  };

  // event
  enum class EventType {
    off_passive,
    off_fault,
    passive_standby,
    passive_fault,
    passive_off,
    standby_active,
    standby_passive,
    standby_fault,
    standby_off,
    active_standstill,
    active_standby,
    active_fault,
    active_off,
    standstill_active,
    standstill_standby,
    standstill_fault,
    standstill_off,
    fault_off,
    fault_passive,
    spin
  };
  struct e_off_passive          {static constexpr auto name{"off_passive"};         static constexpr auto id{EventType::off_passive};         };
  struct e_off_fault            {static constexpr auto name{"off_fault"};           static constexpr auto id{EventType::off_fault};           };
  struct e_passive_standby      {static constexpr auto name{"passive_standby"};     static constexpr auto id{EventType::passive_standby};     };
  struct e_passive_fault        {static constexpr auto name{"passive_fault"};       static constexpr auto id{EventType::passive_fault};       };
  struct e_passive_off          {static constexpr auto name{"passive_off"};         static constexpr auto id{EventType::passive_off};         };
  struct e_standby_active       {static constexpr auto name{"standby_active"};      static constexpr auto id{EventType::standby_active};      };
  struct e_standby_passive      {static constexpr auto name{"standby_passive"};     static constexpr auto id{EventType::standby_passive};     };
  struct e_standby_fault        {static constexpr auto name{"standby_fault"};       static constexpr auto id{EventType::standby_fault};       };
  struct e_standby_off          {static constexpr auto name{"standby_off"};         static constexpr auto id{EventType::standby_off};         };
  struct e_active_standstill    {static constexpr auto name{"active_standstill"};   static constexpr auto id{EventType::active_standstill};   };
  struct e_active_standby       {static constexpr auto name{"active_standby"};      static constexpr auto id{EventType::active_standby};      };
  struct e_active_fault         {static constexpr auto name{"active_fault"};        static constexpr auto id{EventType::active_fault};        };
  struct e_active_off           {static constexpr auto name{"active_off"};          static constexpr auto id{EventType::active_off};          };
  struct e_standstill_active    {static constexpr auto name{"standstill_active"};   static constexpr auto id{EventType::standstill_active};   };
  struct e_standstill_standby   {static constexpr auto name{"standstill_standby"};  static constexpr auto id{EventType::standstill_standby};  };
  struct e_standstill_fault     {static constexpr auto name{"standstill_fault"};    static constexpr auto id{EventType::standstill_fault};    };
  struct e_standstill_off       {static constexpr auto name{"standstill_off"};      static constexpr auto id{EventType::standstill_off};      };
  struct e_fault_off            {static constexpr auto name{"fault_off"};           static constexpr auto id{EventType::fault_off};           };
  struct e_fault_passive        {static constexpr auto name{"fault_passive"};       static constexpr auto id{EventType::fault_passive};       };
  struct e_spin                 {static constexpr auto name{"spin"};                static constexpr auto id{EventType::spin};                };
  
  std::unordered_map<EventType, std::string> event_names = {
    {EventType::off_passive,          std::string(e_off_passive::name)},
    {EventType::off_fault,            std::string(e_off_fault::name)},
    {EventType::passive_standby,      std::string(e_passive_standby::name)},
    {EventType::passive_fault,        std::string(e_passive_fault::name)},
    {EventType::passive_off,          std::string(e_passive_off::name)},
    {EventType::standby_active,       std::string(e_standby_active::name)},
    {EventType::standby_passive,      std::string(e_standby_passive::name)},
    {EventType::standby_fault,        std::string(e_standby_fault::name)},
    {EventType::standby_off,          std::string(e_standby_off::name)},
    {EventType::active_standstill,    std::string(e_active_standstill::name)},
    {EventType::active_standby,       std::string(e_active_standby::name)},
    {EventType::active_fault,         std::string(e_active_fault::name)},
    {EventType::active_off,           std::string(e_active_off::name)},
    {EventType::standstill_active,    std::string(e_standstill_active::name)},
    {EventType::standstill_standby,   std::string(e_standstill_standby::name)},
    {EventType::standstill_fault,     std::string(e_standstill_fault::name)},
    {EventType::standstill_off,       std::string(e_standstill_off::name)},
    {EventType::fault_off,            std::string(e_fault_off::name)},
    {EventType::fault_passive,        std::string(e_fault_passive::name)},
    {EventType::spin,                 std::string(e_spin::name)}
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
      * state<s_off>        + event<e_off_passive>          / [this](){transition_action(EventType::off_passive);}          = state<s_passive>
      , state<s_off>        + event<e_off_fault>            / [this](){transition_action(EventType::off_fault);}            = state<s_fault>
      , state<s_off>        + event<e_spin>                 / [this](){spin_action(StateType::off);}
      , state<s_off>        + sml::on_entry<_>              / [this](){on_entry(StateType::off);}
      , state<s_off>        + sml::on_exit<_>               / [this](){on_exit(StateType::off);}

      , state<s_passive>    + event<e_passive_fault>        / [this](){transition_action(EventType::passive_fault);}        = state<s_fault>
      , state<s_passive>    + event<e_passive_standby>      / [this](){transition_action(EventType::passive_standby);}      = state<s_standby>
      , state<s_passive>    + event<e_passive_off>          / [this](){transition_action(EventType::passive_off);}          = state<s_off>
      , state<s_passive>    + event<e_spin>                 / [this](){spin_action(StateType::passive);}
      , state<s_passive>    + sml::on_entry<_>              / [this](){on_entry(StateType::passive);}
      , state<s_passive>    + sml::on_exit<_>               / [this](){on_exit(StateType::passive);}
      
      , state<s_standby>    + event<e_standby_active>       / [this](){transition_action(EventType::standby_active);}       = state<s_active>
      , state<s_standby>    + event<e_standby_fault>        / [this](){transition_action(EventType::standby_fault);}        = state<s_fault>
      , state<s_standby>    + event<e_standby_passive>      / [this](){transition_action(EventType::standby_passive);}      = state<s_passive>
      , state<s_standby>    + event<e_standby_off>          / [this](){transition_action(EventType::standby_off);}          = state<s_off>
      , state<s_standby>    + event<e_spin>                 / [this](){spin_action(StateType::standby);}
      , state<s_standby>    + sml::on_entry<_>              / [this](){on_entry(StateType::standby);}
      , state<s_standby>    + sml::on_exit<_>               / [this](){on_exit(StateType::standby);}

      , state<s_active>     + event<e_active_standstill>    / [this](){transition_action(EventType::active_standstill);}    = state<s_standstill>
      , state<s_active>     + event<e_active_standby>       / [this](){transition_action(EventType::active_standby);}       = state<s_standby>
      , state<s_active>     + event<e_active_fault>         / [this](){transition_action(EventType::active_fault);}         = state<s_fault>
      , state<s_active>     + event<e_active_off>           / [this](){transition_action(EventType::active_off);}           = state<s_off>
      , state<s_active>     + event<e_spin>                 / [this](){spin_action(StateType::active);}
      , state<s_active>     + sml::on_entry<_>              / [this](){on_entry(StateType::active);}
      , state<s_active>     + sml::on_exit<_>               / [this](){on_exit(StateType::active);}

      , state<s_standstill> + event<e_standstill_active>    / [this](){transition_action(EventType::standstill_active);}    = state<s_active>
      , state<s_standstill> + event<e_standstill_standby>   / [this](){transition_action(EventType::standstill_standby);}   = state<s_standby>
      , state<s_standstill> + event<e_standstill_fault>     / [this](){transition_action(EventType::standstill_fault);}     = state<s_fault>
      , state<s_standstill> + event<e_standstill_off>       / [this](){transition_action(EventType::standstill_off);}       = state<s_off>
      , state<s_standstill> + event<e_spin>                 / [this](){spin_action(StateType::standstill);}
      , state<s_standstill> + sml::on_entry<_>              / [this](){on_entry(StateType::standstill);}
      , state<s_standstill> + sml::on_exit<_>               / [this](){on_exit(StateType::standstill);}

      , state<s_fault>      + event<e_fault_off>            / [this](){transition_action(EventType::fault_off);}            = state<s_off>
      , state<s_fault>      + event<e_fault_passive>        / [this](){transition_action(EventType::fault_passive);}        = state<s_passive>
      , state<s_fault>      + event<e_spin>                 / [this](){spin_action(StateType::fault);}
      , state<s_fault>      + sml::on_entry<_>              / [this](){on_entry(StateType::fault);}
      , state<s_fault>      + sml::on_exit<_>               / [this](){on_exit(StateType::fault);}
    );
    // clang-format on
  }

private:
  void transition_action(EventType event);
  void spin_action(StateType state);
  void on_entry(StateType state);
  void on_exit(StateType state);

  MissionFSMBaseCtx &ctx_;
};

} // namespace mission
} // namespace planning
