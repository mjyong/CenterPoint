/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:52:47
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-03-01 15:35:06
 */
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"
namespace planning {
namespace mission {

MissionFSMBase::MissionFSMBase(MissionFSMBaseCtx &ctx) : ctx_(ctx) {}

void MissionFSMBase::transition_action(EventType event) {
  ctx_.transition_action(event);
}

void MissionFSMBase::spin_action(StateType state) { ctx_.spin_action(state); }

void MissionFSMBase::on_entry(StateType state) {
    ctx_.on_entry(state);
}
void MissionFSMBase::on_exit(StateType state) {
    ctx_.on_exit(state);
}

} // namespace mission
} // namespace planning