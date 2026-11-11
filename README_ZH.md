[English](README.md)

# camera-tool

面向机器人视觉工作站的统一相机预览、手眼标定（**眼在手上 Eye-in-Hand** 与 **眼在手外 Eye-to-Hand**）、离线会话回放、三脚架位移监测与 USB 链路稳定性诊断工具。

## 核心特性

- **通用相机后端抽象**：支持 `realsense`、`orbbec`、`uvc` 与离线 `sim` 后端，内置子进程 SDK 隔离与单设备免序列号自动选择（`serial: ""`）。
- **眼在手上标定（Eye-in-Hand）**：对比 7 种 OpenCV 手眼算法，结合平面标定板 IPPE 姿态消歧、非线性重投影精化（Bundle Adjustment）、法兰/TCP 解耦与桌面/深度平面一致性校验。
- **眼在手外标定（Eye-to-Hand）**：通过腕部相机与固定 RGB-D 相机共视桌面 ArUco 标定板建立坐标链，强制执行独立留出姿态复核与深度比例修正后才允许发布外参。
- **纯离线会话回放**：无需导入任何硬件 SDK 或图形库即可复算历史会话，支持在更换末端执行器后通过 `--tcp-offset` 离线重算并导出外参。
- **现场诊断工具**：提供基于静态背景三维特征的三脚架微动监测（`OK` / `WARN` / `FAIL`）与基于 Linux `sysfs` 的 USB 3.x 协商速率及丢帧统计测试。
- **桌面界面与无界面 CLI**：提供五页面 PySide6 暗色桌面端与脚本化 CLI，通过跨进程文件锁防止同一硬件被并发占用。

## 目录结构

```text
src/camera_tool/
├── core/           # SE(3) 几何变换、末端防撞检查、相机投影畸变模型、ArUco 质量门控与会话读写
├── devices/        # 通用相机后端（realsense, orbbec, uvc, sim）、子进程隔离与机械臂控制接口
├── calibration/    # 眼在手上与眼在手外求解器、预设点位规划与采集工作流
├── diagnostics/    # 三脚架场景基线监测与 Linux sysfs USB 链路速率检查
├── maintenance/    # 腕部内参转换、探针角点标定与外参投影可视化核对
├── services/       # 与 Qt 解耦的任务编排及跨进程设备租约管理
├── ui/             # 五页面 PySide6 桌面界面与 3D 坐标系/点云视图
└── resources/      # 参数化 YAML 配置模板、udev 规则脚本与原生桥接源码
```

## 安装说明

运行环境要求 Linux 与 Python `>= 3.12`：

```bash
# 安装核心包与桌面界面依赖
pip install -e .

# 按需安装硬件 SDK 与测试依赖
pip install -e ".[realsense,orbbec,robot,dev]"
```

- 请保持安装标准 `opencv-python`，不要在同一环境中安装 `opencv-python-headless`。
- 如在 Linux 下使用 Orbbec 相机，可按需安装 USB 权限规则：
  ```bash
  sudo bash src/camera_tool/resources/install_orbbec_udev_rules.sh
  ```

## 快速上手

### 1. 启动桌面端

```bash
camera-tool          # 启动五页面桌面端
camera-tool --sim    # 以离线仿真模式启动桌面端
```

### 2. 设备发现与相机预览

```bash
camera-tool preview --list
camera-tool preview --backend realsense
camera-tool preview --backend orbbec --width 1280 --height 800 --fps 30
camera-tool preview --backend sim --headless
```

### 3. 手眼标定

机械臂控制器 IP 可通过 `--robot-ip`、环境变量 `XARM_IP` 或自定义 YAML 配置文件传入。

```bash
# 眼在手上离线仿真自动标定
camera-tool calibrate --mode eye-in-hand --sim --result result/eye_in_hand.yaml

# 实机只读检查（检查取流、标定板质量与机械臂状态，不发送任何使能或运动指令）
camera-tool calibrate --mode eye-in-hand --preset default --robot-ip <ROBOT_IP> --check

# 眼在手上自动采集并求解（预设模板：default / medium_range / orbbec）
camera-tool calibrate --mode eye-in-hand --preset medium_range --robot-ip <ROBOT_IP>

# 眼在手外完整标定
camera-tool calibrate --mode eye-to-hand --robot-ip <ROBOT_IP>
```

### 4. 离线会话回放与 TCP 偏移更新

```bash
# 离线识别并复算历史标定会话
camera-tool replay runs/eye_in_hand/<SESSION_DIR>

# 更换夹爪或工具后离线覆盖 TCP 偏移（x,y,z,r,p,y，单位 mm,rad）并导出新外参
camera-tool replay runs/eye_in_hand/<SESSION_DIR> \
  --tcp-offset 0,0,172,0,0,0 \
  --result result/eye_in_hand_tcp.yaml
```

### 5. 三脚架位移监测与 USB 稳定性测试

```bash
# 在标定姿态锁定静态环境基线
camera-tool tripod lock --result result/eye_to_hand.yaml

# 单帧无界面位移检查（确认 FAIL 时返回退出码 2）
camera-tool tripod watch --once

# 无界面 USB 链路速率与帧间隔稳定性测试
camera-tool usb-test --backend realsense --duration 1800 --interval 5 --csv usb_stability.csv
```

## 配置模板与本地数据约定

- **内置模板**：位于 `src/camera_tool/resources/configs/`：
  - `eye_in_hand/default.yaml` — 近距 RGB-D 安装模板（对应 `waypoints/close_range.yaml`）。
  - `eye_in_hand/medium_range.yaml` — 中距法兰支架安装模板（对应 `waypoints/medium_range.yaml`）。
  - `eye_in_hand/orbbec.yaml` — `1280x800@30` Orbbec RGB-D 模板。
  - `eye_in_hand/waypoints/profiles.yaml` — 机械臂工作空间构型（`arm_6dof`、`arm_5dof`、`arm_7dof`、`arm_compact_6dof`、`arm_long_6dof`）。
  - `eye_to_hand.yaml` 与 `corners.yaml` — 眼在手外及探针角点标定模板。
- **去硬件硬编码**：所有受控模板中 `serial: ""`、`robot.ip: ""` 均留空；单机连接单个同类型相机时自动识别，多机场景可通过 `--serial <SN>` 或本地自定义 YAML 指定。
- **本地运行目录**：标定原始记录（`runs/`）、导出的外参结果（`result/`）及单机腕部相机内参映射（`intrinsics/wrist_<SERIAL>.json`）均保存在本地工作目录并由 `.gitignore` 忽略，不纳入版本库。

## 运行测试

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```
