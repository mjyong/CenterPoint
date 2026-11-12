/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:52:47
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-03-01 15:35:06
 */
#include "src/runnable/mission/recharging/fsm/recharging_fsm_base.h"
#include "src/runnable/mission/recharging/fsm/recharging_fsm_base_ctx.h"
namespace planning {
namespace mission {

RechargingFSMBase::RechargingFSMBase(RechargingFSMBaseCtx &ctx) : ctx_(ctx) {}

void RechargingFSMBase::transition_action(EventType event) {
  ctx_.transition_action(event);
}

void RechargingFSMBase::spin_action(StateType state) { ctx_.spin_action(state); }

void RechargingFSMBase::on_entry(StateType state) {
    ctx_.on_entry(state);
}
void RechargingFSMBase::on_exit(StateType state) {
    ctx_.on_exit(state);
}

} // namespace mission
} // namespace planning