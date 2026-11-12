#!/bin/bash
filename=/user_name.txt
HOST_USER_NAME=$(cat ${filename})
if [ -z "${HOST_USER_NAME}" ];then
    echo "${filename} file not found"
    exit 1
fi
if [ "${HOST_USER_NAME}" == "$(id -un)" ];then
    echo "you have already in ${HOST_USER_NAME} user"
    exit 0
fi
su ${HOST_USER_NAME} -c "script -q -c /bin/bash /dev/null"