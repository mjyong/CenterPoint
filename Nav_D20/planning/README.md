# planning

机器人项目PNC仓库。D20 上和感知、定位怎么接、怎么在本机把节点拉起来，见 [docs/ONBOARDING.md](docs/ONBOARDING.md)。

## Build
docker 镜像版本: 10.42.11.106:5000/jetpack6.1-crosscompile-aarch64-x86:v2.2.1

编译脚本: build.sh
```shell
build.sh -d  <addr>         download and install to 192.168.<addr>
           -arm                       build arm
           -x86                       build x86
           -p <project>               package project (e.g., d10, d20, r11)
```
注意：编译依赖seres_interface仓库，编译脚本先编译seres_interface仓库，默认路径为../seres_interface/
file: build.sh
line: 20 && 43
```
    colcon build --base-paths ../seres_interface/ \
        --packages-select nav2_interface mc_interface 
```