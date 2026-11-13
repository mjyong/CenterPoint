#!/bin/bash
start_time=$(date +%s)

# 定义参数
VERSION_NUMBER="1.0.1"               # 版本号参数
PACKAGE_NAME="seres_interface"       # 软件包名称，涉及软件安装目录
DEB_PACKAGE_NAME="seres-interface"   # deb信息中的包名称

# 检查环境变量或设置默认值
if [ -z "${ROBOT_TYPE}" ]; then
    ROBOT_TYPE="R11"
fi

if [[ -z "$CI_COMMIT_REF_NAME" || -z "$CI_COMMIT_SHA" ]]; then
  echo "\u8bf7\u8bbe\u7f6e CI_COMMIT_REF_NAME \u548c CI_COMMIT_SHA \u73af\u5883\u53d8\u91cf\u3002"
  CI_COMMIT_REF_NAME=$(git branch --show-current)
  CI_COMMIT_SHA=$(git log -1 --format=%H)
fi

declare -A archs
archs["x86"]="amd64"
archs["aarch"]="arm64"
archs["aarch64"]="arm64"
archs["arm64"]="arm64"
archs["x86_aarch"]="arm64"

ARCH=""

# 获取当前架构和时间戳
if [ -z "${CI_ARCH}" ] || [[ ! ${archs[$CI_ARCH]} ]]; then
    ARCH="arm64"
else
    ARCH=${archs[${CI_ARCH}]}
fi

TIMESTAMP=$(date +"%Y%m%d%H")
CI_COMMIT_REF_NAME="${CI_COMMIT_REF_NAME##*/}"
VERSION_STRING=v${VERSION_NUMBER}-${TIMESTAMP}
DEB_DIR="build_tmps/${PACKAGE_NAME}-${ROBOT_TYPE}-Humble-${VERSION_STRING}-${ARCH}"
echo "deb包命名为 ${DEB_DIR}"

# 创建目录结构
mkdir -p "${DEB_DIR}/opt/seres/middleware/${PACKAGE_NAME}"

if [[ -d "install" ]]; then
  cp -r install/* ${DEB_DIR}/opt/seres/middleware/${PACKAGE_NAME}
  rm -rf ${DEB_DIR}opt/seres/middleware/${PACKAGE_NAME}/include
  rm -rf ${DEB_DIR}opt/seres/middleware/${PACKAGE_NAME}/bin
else
  echo "未找到 install/ 目录，退出。"
  rm -rf "build_tmps"
  exit 1
fi

# 创建 version.txt
cat <<EOF >"${DEB_DIR}/opt/seres/middleware/${PACKAGE_NAME}/version.txt"
date: $(date +"%Y-%m-%d %H:%M:%S")
sha: ${CI_COMMIT_SHA:0:6}
branch: ${CI_COMMIT_REF_NAME}
EOF

# 创建 DEBIAN 控制文件
mkdir -p "${DEB_DIR}/DEBIAN"

cat <<EOF >"${DEB_DIR}/DEBIAN/control"
Package: ${DEB_PACKAGE_NAME}
Version: ${VERSION_NUMBER}
Architecture: ${ARCH}
Maintainer: <yunxing.wang@seres.cn>
Description: ${PACKAGE_NAME}. Installs to /opt/seres/middleware/
EOF

cat <<EOF > "${DEB_DIR}/DEBIAN/preinst"
#!/bin/bash

EOF

cat <<EOF > "${DEB_DIR}/DEBIAN/postinst"
#!/bin/bash

# Identify the executing user
EXEC_USER="root"
EXEC_GROUP="root"

chmod -R 777 /opt/seres/middleware
chmod -R 777 /opt/seres/middleware/${PACKAGE_NAME}

sudo sh -c "echo 'source /opt/seres/middleware/${PACKAGE_NAME}/setup.bash >/dev/null 2>&1' >> /etc/profile.d/seres_env.sh"
sudo sh -c "echo '/opt/seres/middleware/${PACKAGE_NAME}/lib' >> /etc/ld.so.conf.d/seres.conf"
echo "Running ldconfig..."
/sbin/ldconfig -v >/var/log/seres_ldconfig.log 2>&1

echo "安装完成。"
echo "执行用户: \${EXEC_USER}, 用户组: \${EXEC_GROUP}"
EOF

cat ${DEB_DIR}/opt/seres/middleware/${PACKAGE_NAME}/version.txt || true

# 设置 postinst 可执行权限
chmod  755 "${DEB_DIR}/DEBIAN/postinst"
chmod  755 "${DEB_DIR}/DEBIAN/preinst"
chmod  644 "${DEB_DIR}/DEBIAN/control"
find "${DEB_DIR}" -type d -exec chmod 755 {} \;

# 构建 .deb 包
echo "DEB_DIR: ${DEB_DIR}"
dpkg-deb -Zgzip -z6 --build "${DEB_DIR}"

if [ ! -f "${DEB_DIR}.deb" ]; then
   echo "构建失败"
   rm -rf "build_tmps"
   exit 1
fi

RELEASE_PATH=""
if [[ -d "${CI_OUTPUT_PATH}" ]]; then
    RELEASE_PATH="${CI_OUTPUT_PATH}/${ARCH}/"
    mkdir -p ${RELEASE_PATH}
else
    RELEASE_PATH="./release/${ARCH}"
fi

# 创建发布目录并移动deb包
mkdir -p "${RELEASE_PATH}"
rm -rf "${RELEASE_PATH}"/*

cp "${DEB_DIR}.deb" "${RELEASE_PATH}/"
rm -rf "build_tmps"

# 提示结果
echo "构建完成，deb 包位于 ${RELEASE_PATH}/${DEB_DIR}.deb"

end_time=$(date +%s)
elapsed_time=$(($end_time - $start_time))
echo "total cost time of compile: $elapsed_time s"
