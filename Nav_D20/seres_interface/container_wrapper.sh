#!/bin/bash

set -e

echo "CI_COMMIT_SHA: ${CI_COMMIT_SHA}"
echo "CI_PIPELINE_ID: ${CI_PIPELINE_ID}"
echo "CI_PROJECT_NAME: ${CI_PROJECT_NAME}"
echo "CI_COMMIT_REF_NAME: ${CI_COMMIT_REF_NAME}"
echo "CI_JOB_NAME: ${CI_JOB_NAME}"

curdir=$(dirname "$(realpath "$0")")

packages=""
fail_continue=""
clean=""

print_help() {
    length=${#1}
    spaces=$(printf "%*s" "$length" "")
    echo "$1 [--arch <x86 | aarch | x86_aarch>] 指定目标产物CPU架构，不指定默认交叉编译"
    echo "${spaces} [--fail-continue | -f] 允许某些包失败后继续编译"
    echo "${spaces} [--help | -h] 查看帮助信息"
    echo "${spaces} [--packages-select | -p <package1 packages2...>] 指定某些包编译，至少一个包"
    echo "${spaces} [--clean | --clear | -c] 清除所有编译产物"
    echo "${spaces} [--run | --r ] docker 执行指定脚本"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            arch=$2
            shift 2
            ;;
        --fail-continue | -f)
            fail_continue="-f"
            shift
            ;;
        --help | -h)
            print_help $0
            exit 0
            ;;
        --packages-select | -p)
            shift
            packages="-p $@"
            shift $#
            ;;
        --clean | --clear | -c)
            clean="-c"
            shift
            ;;
        --run | -r)
            run=$2
            shift 2
            ;;
        *)
            echo "illegal input $*"
            print_help $0
            exit 1
            ;;
    esac
done

declare -A containers
containers["x86"]="robot_ubuntu20_foxy_x86_aarch"
containers["aarch"]="robot_ubuntu20_foxy_aarch"
containers["aarch64"]="robot_ubuntu20_foxy_aarch"
containers["x86_aarch"]="robot_ubuntu20_foxy_x86_aarch"
containers["humble_x86_aarch"]="robot_ubuntu22_humble_x86_aarch"
containers["humble_x86"]="robot_ubuntu22_humble_x86_aarch"
containers["arm_web_build"]="arm_web_build"


declare -A sysroots
sysroots["x86"]="/"
sysroots["aarch"]="/"
sysroots["aarch64"]="/"
sysroots["x86_aarch"]="~/cc_ws/sysroot_docker"
sysroots["humble_x86_aarch"]="~/cc_ws/sysroot_docker"
sysroots["humble_x86"]="/"

if [ ! -z ${CI_ARCH} ]; then
    arch=${CI_ARCH}
fi

if [ -z ${arch} ]; then
    arch="humble_x86"
    arch="${arch%_x64}"
fi

if [[ ! ${containers[$arch]} ]]; then
    echo "${arch} not in ${!containers[@]}"
    exit 1
fi

if ! docker ps --filter "name=${containers[$arch]}" --format '{{.Names}}' | grep -w "${containers[$arch]}" > /dev/null; then
    echo "Container is not running, starting it..."
    docker start ${containers[$arch]}
else
    echo "Container is already running."
fi
# 不空,代表 deploy.sh 来进行调用...
if [ -n "$run" ]; then
    echo "get $run,will run deploy"
    docker exec -i ${containers[$arch]} bash -c \
        "cd ${curdir} && \
        chmod -R a+w $(pwd);$run; \
        run_exit_code=\$?; \
        echo \"run_exit_code: xx\$run_exit_code xx\"; \
        mkdir -p  build/ install/ log/ checkans/ tempcpp/; \
        chmod -R 777 src/module/robot_webctr/ build/ install/ log/ checkans/ tempcpp/; \
        chmod -R a+w \$(git submodule status | awk '{print \$2}'); \
        exit \$run_exit_code"
    exit $?
fi
echo "docker exec -it ${containers[$arch]} bash -c"
echo "cd ${curdir};./build.sh --sysroot ${sysroots[$arch]}"
echo "CI_PROJECT_DIR: $(pwd)"
docker exec -i ${containers[$arch]} bash -c \
    "export TERM=xterm-256color && \
    export MAKEFLAGS='--color=always' && \
    cd ${curdir} && \
    script -q -c 'chmod -R a+w $(pwd);./build.sh --arch ${arch} --sysroot ${sysroots[$arch]} \
    ${fail_continue} ${packages} ${clean};echo \$? > build/build_exit_code.txt' /dev/null; \
    build_exit_code=\$(cat build/build_exit_code.txt);echo \"build_exit_code: xx\${build_exit_code}xx\"; \
    mkdir -p  build/ install/ log/ ; \
    chmod -R 777 build/ install/ log/ ; \
    chmod -R a+w \$(git submodule status | awk '{print \$2}'); \
    exit \$build_exit_code"


