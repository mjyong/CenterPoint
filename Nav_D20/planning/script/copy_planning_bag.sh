#!/bin/bash

#REMOTE_HOST="sunrise@10.0.0.35"
REMOTE_HOST="sunrise@192.168.99.10"
#REMOTE_HOST="nvidia@192.168.122.100"
#REMOTE_PASSWORD="nvidia"
REMOTE_PASSWORD="sunrise"
REMOTE_DIR="/tmp"
LOCAL_DIR="./planning_bag" #本地存储路径
EXEC_NAME="planning"

# 创建本地目录
mkdir -p "$LOCAL_DIR"

echo "搜索最新文件夹..."
LATEST_FOLDER=$(sshpass -p "$REMOTE_PASSWORD" ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && ls -td '$EXEC_NAME'_* 2>/dev/null | head -1")

if [ -z "$LATEST_FOLDER" ]; then
    echo "未找到 '$EXEC_NAME'_* 开头的文件夹"
    # 尝试查找其他模式
    LATEST_FOLDER=$(sshpass -p "$REMOTE_PASSWORD" ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && ls -td *_202*-*-*-*-*_* 2>/dev/null | head -1")
    
    if [ -z "$LATEST_FOLDER" ]; then
        echo "错误: 未找到任何匹配的文件夹"
        exit 1
    fi
fi

echo "找到: $LATEST_FOLDER"

# 拷贝前先获取远程文件夹大小
REMOTE_SIZE=$(sshpass -p "$REMOTE_PASSWORD" ssh "$REMOTE_HOST" "du -sh '$REMOTE_DIR/$LATEST_FOLDER' 2>/dev/null | cut -f1")
echo "远程文件夹大小: ${REMOTE_SIZE:-未知}"

# 远程有 rsync 则用 rsync 显示实时总进度，否则 scp + 本地大小监控
if sshpass -p "$REMOTE_PASSWORD" ssh "$REMOTE_HOST" "command -v rsync" >/dev/null 2>&1; then
    COPY_CMD="rsync"
else
    COPY_CMD="scp"
fi

# $1: 远程源路径  $2: 本地目标  $3: 用于大小监控的本地路径
copy_with_progress() {
    if [ "$COPY_CMD" = "rsync" ]; then
        rsync -rh --info=progress2 --partial \
            -e "sshpass -p $REMOTE_PASSWORD ssh" \
            "$REMOTE_HOST:$1" "$2"
    else
        (
            while sleep 2; do
                if [ -e "$3" ]; then
                    printf '\r已拷贝: %-10s / %s' \
                        "$(du -sh "$3" 2>/dev/null | cut -f1)" "${REMOTE_SIZE:-?}"
                fi
            done
        ) &
        local monitor_pid=$!
        sshpass -p "$REMOTE_PASSWORD" scp -r "$REMOTE_HOST:$1" "$2"
        local scp_ret=$?
        kill "$monitor_pid" 2>/dev/null
        wait "$monitor_pid" 2>/dev/null
        printf '\r\033[K'
        return $scp_ret
    fi
}

# 修复旧版本(如 Humble)录制的 bag 在 Jazzy 下 rqt_bag 回放报
# IndexError: offered_qos_profiles[0] 的问题:
# 旧版录制时 QoS 未写入元数据, reindex 后为空列表, Jazzy 的
# rqt_bag 取 offered_qos_profiles[0] 越界。为所有 topic 补默认 QoS。
fix_qos_profiles() {
    python3 - "$1" <<'PYEOF'
import sys, yaml

bag_dir = sys.argv[1]
meta_path = bag_dir.rstrip('/') + '/metadata.yaml'
try:
    with open(meta_path) as f:
        d = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit(0)

qos = {'history': 1, 'depth': 10, 'reliability': 1, 'durability': 2,
       'deadline': {'sec': 0, 'nsec': 0}, 'lifespan': {'sec': 0, 'nsec': 0},
       'liveliness': 0, 'liveliness_lease_duration': {'sec': 0, 'nsec': 0},
       'avoid_ros_namespace_conventions': False}

fixed = 0
for t in d.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', []):
    tm = t.get('topic_metadata', {})
    if not tm.get('offered_qos_profiles'):
        tm['offered_qos_profiles'] = [qos]
        fixed += 1

if fixed > 0:
    with open(meta_path, 'w') as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print(f"已为 {fixed} 个 topic 补充默认 QoS (rqt_bag 兼容)")
PYEOF
}

echo "开始拷贝..."
copy_with_progress "$REMOTE_DIR/$LATEST_FOLDER" "$LOCAL_DIR/" "$LOCAL_DIR/$LATEST_FOLDER"
copy_with_progress "$REMOTE_DIR/$EXEC_NAME.INFO" "$LOCAL_DIR/$LATEST_FOLDER/" "$LOCAL_DIR/$LATEST_FOLDER/$EXEC_NAME.INFO"
ros2 bag reindex $LOCAL_DIR/$LATEST_FOLDER
fix_qos_profiles "$LOCAL_DIR/$LATEST_FOLDER"

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "✓ 成功拷贝文件夹: $LATEST_FOLDER"
    echo "本地路径: $LOCAL_DIR/$LATEST_FOLDER"
    echo "文件夹大小: $(du -sh "$LOCAL_DIR/$LATEST_FOLDER" | cut -f1)"
    echo "======================================"
else
    echo "✗ 拷贝失败，请检查:"
    echo "1. SSH连接是否正常"
    echo "2. 远程路径是否正确: $REMOTE_HOST:$REMOTE_DIR/$LATEST_FOLDER"
    echo "3. 是否有足够的权限"
    exit 1
fi