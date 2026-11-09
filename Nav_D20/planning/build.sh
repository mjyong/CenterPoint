#!/bin/bash

work_dir="$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$work_dir"
project_name=$(basename $work_dir)
temp_name=.temp
package_dir=$work_dir/$temp_name/$project_name
cpu_num=`cat /proc/stat |grep cpu[0-9] -c`
cpu_num=$[$cpu_num-4]
echo "cpu_num: $cpu_num"
ssh_options="-o StrictHostKeyChecking=no"
remote_user="nvidia"
remote_pwd="nvidia"

function build_arm() {
    basepath=$(pwd)
    echo $basepath
    unset AMENT_PREFIX_PATH
    source /l4t/jetson_arm64_sysroot/opt/ros/humble/setup.bash
    colcon build \
        --base-paths ../seres_interface/ \
        --packages-select nav2_interface common_interface \
            mc_interface perception_interface rte_interface \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_TOOLCHAIN_FILE=$basepath/toolchain-aarch64.cmake
    echo "build seres_interface done! building woosh_robot_msgs..."
    sleep 1
    colcon build --base-paths third_party/ \
        --packages-select woosh_robot_msgs \
        --cmake-force-configure \
        --cmake-args \
            -DCMAKE_TOOLCHAIN_FILE=$basepath/toolchain-aarch64.cmake
    echo "build woosh_robot_msgs done! building planning..."
    sleep 1
    source install/setup.bash

    HOST_PROTOC=$(which protoc)
    if [ -z "$HOST_PROTOC" ]; then
        log_error "Host protoc not found. Please install protobuf-compiler."
        exit 1
    fi
    echo "Using host protoc: $HOST_PROTOC"

    colcon build \
        --packages-select planning \
        --cmake-force-configure \
        --cmake-args \
            -DProtobuf_PROTOC_EXECUTABLE=${HOST_PROTOC} \
            -DCMAKE_TOOLCHAIN_FILE=$basepath/toolchain-aarch64.cmake        
    echo "build planning done!"
    echo "arm64" > $basepath/.build_arch
    unset AMENT_PREFIX_PATH
}

function build_x86() {
    basepath=$(pwd)
    echo $basepath
    source /opt/ros/humble/setup.bash
    colcon build --base-paths ../seres_interface/ \
        --packages-select nav2_interface common_interface \
            mc_interface perception_interface rte_interface
    echo "build seres_interface done! building woosh_robot_msgs..."
    sleep 1
    colcon build --base-paths third_party/  --packages-select woosh_robot_msgs
    echo "build woosh_robot_msgs done! building planning..."
    sleep 1
    source install/setup.bash
    export PATH=/usr/local/cuda/bin/:$PATH
    colcon build --packages-select planning
    echo "build planning done!"
    echo "amd64" > $basepath/.build_arch
}


function package() {
    local project="$1"
    basepath=$(pwd)
    echo $basepath
    echo "package planning..."

    if [ -z "${project}" ]; then
        echo "Error: project parameter is required (e.g., d10, d20, r11)"
        exit 1
    fi

    local project_lower=$(echo "${project}" | tr '[:upper:]' '[:lower:]')
    local project_upper=$(echo "${project}" | tr '[:lower:]' '[:upper:]')
    local vin_path="${project_lower}/001"

    local VERSION_NUMBER="1.0.1"
    local PACKAGE_NAME="planning"
    local DEB_PACKAGE_NAME="planning"
    rm -rf $basepath/*.deb

    robot_type="${project_upper}"

    if [[ -z "$CI_COMMIT_REF_NAME" || -z "$CI_COMMIT_SHA" ]]; then
        CI_COMMIT_REF_NAME=$(git branch --show-current 2>/dev/null || echo "unknown")
        CI_COMMIT_SHA=$(git log -1 --format=%H 2>/dev/null || echo "unknown")
    fi

    declare -A archs
    archs["x86"]="amd64"
    archs["aarch"]="arm64"
    archs["aarch64"]="arm64"
    archs["arm64"]="arm64"
    archs["x86_aarch"]="arm64"

    local ARCH=""
    if [ -f "$basepath/.build_arch" ]; then
        ARCH=$(cat $basepath/.build_arch)
        echo "Detected build architecture from .build_arch: $ARCH"
    elif [ -z "${CI_ARCH}" ] || [[ ! ${archs[$CI_ARCH]} ]]; then
        ARCH="amd64"
    else
        ARCH=${archs[${CI_ARCH}]}
    fi

    local TIMESTAMP=$(date +"%Y%m%d%H")
    CI_COMMIT_REF_NAME="${CI_COMMIT_REF_NAME##*/}"
    local VERSION_STRING=v${VERSION_NUMBER}-${TIMESTAMP}
    local DEB_DIR="build_tmps/${PACKAGE_NAME}-${robot_type}-Humble-${VERSION_STRING}-${ARCH}"
    echo "deb package name: ${DEB_DIR}"

    mkdir -p "${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}"

    if [[ -d "install" ]]; then
        cp -r install/* ${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}

        local vin_file="${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}/planning/lib/planning/robot_factory/default_robot_vin"
        if [ -f "${vin_file}" ]; then
            echo "${vin_path}" > "${vin_file}"
            echo "Set default_robot_vin to: ${vin_path}"
        else
            echo "Warning: default_robot_vin not found at ${vin_file}"
        fi

        if [[ "${ARCH}" == "arm64" ]]; then
            find "${DEB_DIR}" -type f \( -name "setup.bash" -o -name "local_setup.bash" -o -name "local_setup.sh" -o -name "setup.sh" -o -name "_local_setup_util_sh.py" -o -name "*.cmake" \) -exec sed -i 's|/l4t/jetson_arm64_sysroot/opt/ros/humble|/opt/ros/humble|g' {} \;
        fi
    else
        echo "install/ directory not found, abort."
        rm -rf "build_tmps"
        exit 1
    fi

    cat <<EOF >"${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}/version.txt"
date: $(date +"%Y-%m-%d %H:%M:%S")
sha: ${CI_COMMIT_SHA:0:6}
branch: ${CI_COMMIT_REF_NAME}
EOF

    mkdir -p "${DEB_DIR}/DEBIAN"

    cat <<EOF >"${DEB_DIR}/DEBIAN/control"
Package: ${DEB_PACKAGE_NAME}
Version: ${VERSION_NUMBER}
Architecture: ${ARCH}
Maintainer: <tian.zhao@seres.cn>
Description: ${PACKAGE_NAME}. Installs to /opt/seres/nav/${PACKAGE_NAME}
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/preinst"
#!/bin/bash
if [ -d "/opt/seres/nav/${PACKAGE_NAME}" ]; then
    systemctl stop ${PACKAGE_NAME}.service 2>/dev/null || true
    systemctl disable ${PACKAGE_NAME}.service 2>/dev/null || true
    rm -f /etc/systemd/system/${PACKAGE_NAME}.service
    systemctl daemon-reload
    rm -rf /opt/seres/nav/${PACKAGE_NAME}
fi
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/postinst"
#!/bin/bash
cat > /etc/systemd/system/${PACKAGE_NAME}.service << SVCEOF
[Unit]
Description=seres planning service
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/seres/nav/${PACKAGE_NAME}
ExecStart=/bin/bash -c 'source /opt/seres/nav/${PACKAGE_NAME}/setup.bash; ros2 run planning planning'
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=planning

[Install]
WantedBy=multi-user.target
SVCEOF
systemctl daemon-reload
systemctl enable ${PACKAGE_NAME}.service
systemctl start ${PACKAGE_NAME}.service
EOF

    cat <<EOF >"${DEB_DIR}/DEBIAN/prerm"
#!/bin/bash
systemctl stop ${PACKAGE_NAME}.service 2>/dev/null || true
systemctl disable ${PACKAGE_NAME}.service 2>/dev/null || true
rm -f /etc/systemd/system/${PACKAGE_NAME}.service
systemctl daemon-reload
rm -rf /opt/seres/nav/${PACKAGE_NAME}
EOF

    cat ${DEB_DIR}/opt/seres/nav/${PACKAGE_NAME}/version.txt || true

    chmod 755 "${DEB_DIR}/DEBIAN/postinst"
    chmod 755 "${DEB_DIR}/DEBIAN/preinst"
    chmod 755 "${DEB_DIR}/DEBIAN/prerm"
    chmod 644 "${DEB_DIR}/DEBIAN/control"
    find "${DEB_DIR}" -type d -exec chmod 755 {} \;

    echo "DEB_DIR: ${DEB_DIR}"
    dpkg-deb -Zgzip -z6 --build "${DEB_DIR}"

    if [ ! -f "${DEB_DIR}.deb" ]; then
        echo "deb build failed"
        rm -rf "build_tmps"
        exit 1
    fi

    cp "${DEB_DIR}.deb" "$basepath"
    rm -rf "build_tmps"

    echo "deb package built at $basepath"
    echo "package done."
}

function download_and_install() {
    local host=$1
    if [ -z "$host" ]; then
        echo "Error: target host not specified"
        exit 1
    fi

    local deb_file=$(ls -t ${work_dir}/*.deb 2>/dev/null | head -1)
    if [ -z "$deb_file" ]; then
        echo "Error: no .deb file found in ${work_dir}"
        exit 1
    fi

    local deb_name=$(basename $deb_file)
    local remote_tmp="/tmp/${deb_name}"

    echo "Deploying ${deb_name} to ${remote_user}@${host}..."

    sshpass -p "${remote_pwd}" scp ${ssh_options} "${deb_file}" "${remote_user}@${host}:${remote_tmp}"
    if [ $? -ne 0 ]; then
        echo "Error: scp failed"
        exit 1
    fi

    sshpass -p "${remote_pwd}" ssh ${ssh_options} "${remote_user}@${host}" \
        "echo '${remote_pwd}' | sudo -S dpkg -i ${remote_tmp} && rm -f ${remote_tmp}"
    if [ $? -ne 0 ]; then
        echo "Error: dpkg install failed"
        exit 1
    fi

    echo "Deployment to ${host} completed"
}

function print_usage() {
    echo "build.sh -d  <addr>         download and install to 192.168.<addr>
           -arm                       build arm
           -x86                       build x86
           -p <project>               package project (e.g., d10, d20, r11)"
}

function main() {
    if [ $# -lt 1 ]; then
        print_usage
        exit
    fi
    
    platform=$1

    case $platform in
    -arm)
        build_arm $@
        ;;
    -x86)
        build_x86 $@
        ;;
    -p|-package)
        if [ $# -lt 2 ]; then
            echo "Error: -p requires a project parameter (e.g., d10, d20, r11)"
            print_usage
            exit 1
        fi
        package "$2"
        ;;
    -d|-download)
        if [ $# -lt 2 ]; then
            print_usage
            exit 1
        fi
        download_and_install "192.168.$2"
        ;;
    *)
        print_usage
        ;;
    esac
}

main $@
