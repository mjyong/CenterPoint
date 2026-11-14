#!/bin/bash
# Where the replay scripts take the D20 stack from (sourced by run_planning_stack_debug.sh, replay_planning_check.sh).
# - NAV_STACK_ROOT=<dir> (set by Nav_D20/docker/run.sh in the nav-dev image): /opt/ros/humble plus one prefix per
#   module, <dir>/{nav_perception,slam,perception,planning}, built by Nav_D20/docker/build_stack.sh each against its
#   own interface copies, as on the dog. Each process runs with its own module's prefix in front (nav_exec); the
#   command-line tools (bag play / record, topic pub) see all four, planning's last so its messages win.
# - otherwise the micromamba environment /public/zjst/robot/planning_humble_env, as before.
NAV_LEGACY_PREFIX=/public/zjst/robot/planning_humble_env
NAV_MODULES=(nav_perception slam perception planning)

nav_env_setup() {
  if [[ -n "${NAV_STACK_ROOT:-}" ]]; then
    local m
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    for m in "${NAV_MODULES[@]}"; do
      # shellcheck disable=SC1090
      [[ -f "${NAV_STACK_ROOT}/${m}/setup.bash" ]] && source "${NAV_STACK_ROOT}/${m}/setup.bash"
    done
    set -u
  else
    export MAMBA_ROOT_PREFIX="${NAV_LEGACY_PREFIX}/root"
    export CONDA_BUILD=0
    set +u
    eval "$("${NAV_LEGACY_PREFIX}/micromamba" shell hook -s bash)"
    micromamba activate "${NAV_LEGACY_PREFIX}/env"
    # shellcheck disable=SC1091
    source "${NAV_LEGACY_PREFIX}/colcon/install/setup.bash"
    set -u
  fi
}

# nav_bin <module> <package> <executable>: where the executable is installed
nav_bin() {
  if [[ -n "${NAV_STACK_ROOT:-}" ]]; then
    echo "${NAV_STACK_ROOT}/$1/$2/lib/$2/$3"
  else
    echo "${NAV_LEGACY_PREFIX}/colcon/install/$2/lib/$2/$3"
  fi
}

# nav_exec <module> <command...>: exec the command with that module's prefix in front (its own interface copies win);
# use in a subshell: ( nav_exec slam "${loc_bin}" ) > log 2>&1 &
nav_exec() {
  local m=$1
  shift
  if [[ -n "${NAV_STACK_ROOT:-}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${NAV_STACK_ROOT}/${m}/setup.bash"
    set -u
  fi
  exec "$@"
}
