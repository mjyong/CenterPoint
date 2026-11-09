#!/bin/bash

REMOTE_HOST="nvidia@192.168.122.100"
REMOTE_PASSWORD="nvidia"
REMOTE_DIR="/tmp"
LOCAL_DIR="./plannig_bag" #本地存储路径
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
echo "开始拷贝..."

# 显示进度
echo "正在从远程拷贝数据包"
sshpass -p "$REMOTE_PASSWORD" scp -r "$REMOTE_HOST:$REMOTE_DIR/$LATEST_FOLDER" "$LOCAL_DIR/"
echo "正在从远程拷贝日志文件"
sshpass -p "$REMOTE_PASSWORD" scp -r "$REMOTE_HOST:$REMOTE_DIR/'$EXEC_NAME'.INFO" "$LOCAL_DIR/$LATEST_FOLDER"

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