#!/bin/bash
echo "usage:bash $0"
repository=10.42.11.106:5000/jetpack6.1-crosscompile-aarch64-x86
tag=latest
proj_dir=$(dirname "$(realpath $0)")
cd ${proj_dir}
proj_name=$(basename ${proj_dir})
docker_code_dir=/app_code/${proj_name}
HOST_GID=$(id -g)
HOST_UID=$(id -u)
HOST_USR=$(id -un)
cmd_str="groupadd -g ${HOST_GID} ${HOST_USR} && useradd -u ${HOST_UID} -g ${HOST_GID} -m ${HOST_USR} && echo ${HOST_USR} > /user_name.txt && \"bash\""
echo "cmd_str:${cmd_str}"
docker run --rm -it -v ${proj_dir}:${docker_code_dir} -w ${docker_code_dir} ${repository}:${tag} bash -c "${cmd_str}"