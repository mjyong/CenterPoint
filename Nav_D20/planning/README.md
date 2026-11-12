# planning

机器人项目PNC仓库

## Build

docker 镜像版本: 10.42.11.106:5000/jetpack6.1-crosscompile-aarch64-x86:v2.2.1

编译脚本: build.sh

```shell
build.sh -d  <addr>         download and install to 192.168.<addr>
           -arm                       build arm (set DISABLE_CUDA=1 to build without CUDA)
           -x86                       build x86
           -p <project>               package project (e.g., d10, d20, r11)

DISABLE_CUDA=1 bash build.sh -arm    # CPU-only 构建， d30项目使用
bash build.sh -arm # with CUDA 构建
```

注意：编译依赖seres\_interface仓库，编译脚本先编译seres\_interface仓库，默认路径为../seres\_interface/
file: build.sh
line: 20 && 43

```
    colcon build --base-paths ../seres_interface/ \
        --packages-select nav2_interface mc_interface 
```

