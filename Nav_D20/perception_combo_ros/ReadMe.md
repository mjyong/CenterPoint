### 1. 编译
1. `bash enter_docker_env.sh` — 进入 Docker 环境
2. `bash switch_user.sh` — 切换用户环境
3. `bash build_perception_crosscompile.sh` — 交叉编译感知代码，产物在 `install/`
4. `run_perception.sh` — 通过 tmux 挂在后台运行感知节点（开发机本地调试）
5. `run_lidar_seg.sh` — 运行单一节点示例脚本

### 2. 打包
1. `./deploy_perception.sh -v 0.1` — 在编译产物 `install/` 目录旁执行，打包所有可执行程序和配置
2. 将生成的 `perception-D20-02-v0.00.01.xxxxxxxx-arm64.deb` 传输到 Orin

### 3. 安装与运行
1. **安装 deb 包**  
   ```bash
   sudo dpkg -i perception-D20-02-v0.00.01.xxxxxxxx-arm64.deb
   ```
   安装后 postinst 自动完成：  
   - 生成 `/etc/systemd/system/nav_perception.service`（替换运行用户）  
   - `systemctl enable nav_perception.service`（设为开机自启）

2. **手动启动服务**  
   ```bash
   sudo systemctl start nav_perception.service
   ```

3. **查看运行状态**  
   ```bash
   sudo systemctl status nav_perception.service
   journalctl -u nav_perception.service -f      # 实时日志
   ```

4. **调试模式（tmux 可视化）**  
   ```bash
   # 先停止 service 再用 tmux，避免重复启动
   sudo systemctl stop nav_perception.service
   
   bash /opt/seres/perception/start/run_perception_tmux_opt.sh
   ```

5. **开机自启验证**  
   ```bash
   sudo systemctl is-enabled nav_perception.service   # 应输出 enabled
   ```

### 4. 清除安装内容
1. **停止并卸载**  
   ```bash
   sudo dpkg -r nav-perception-d20-02           # 触发 prerm + postrm 钩子
   # prerm 自动停止 nav_perception.service 并 disable
   # postrm 自动删除 /etc/systemd/system/nav_perception.service 并 daemon-reload
   ```

2. **完全清除（含配置文件）**  
   ```bash
   sudo dpkg --purge nav-perception-d20-02 
   ```

3. **手动清理感知目录**（dpkg -r 不会删除 `/opt/seres/perception/`，需手动）  
   ```bash
   sudo rm -rf /opt/seres/nav_perception
   ```

4. **确认清除**  
   ```bash
   dpkg -l | grep nav-perception-d20-02             # 应无输出
   systemctl status nav_perception.service    # 应提示 Unit not found
   ```

