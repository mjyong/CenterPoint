USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)
LOCAL_HOST=`hostname`
DOCKER_HOME="/home/$USER"

if [ "$USER" == "root" ];then
    DOCKER_HOME="/root"
fi
if [ ! -d "$HOME/.cache" ];then
    mkdir "$HOME/.cache"
fi

IMG="10.42.11.106:5000/jetpack6.1-crosscompile-aarch64-x86:v2.2.1"

CONTAINER_NAME=ubuntu_jammy_x86

#WORK_DIR: Modify according to your project catalog
WORK_DIR=/home/$USER/works
DST_DIR=/userdata

xhost +local:root
docker run -it   \
	-d \
	--gpus all \
	--net host \
	--env="DISPLAY"   \
	--env="QT_X11_NO_MITSHM=1" \
	--env="ROS_DOMAIN_ID=22" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  \
	--name $CONTAINER_NAME \
	-v $HOME/.cache:/root/.cache \
	-v ${WORK_DIR}:${DST_DIR} \
	-w /userdata \
	$IMG \
	/bin/bash


