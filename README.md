[中文说明](README_ZH.md)

# camera-tool

Unified camera preview, Hand-Eye calibration (**Eye-in-Hand** & **Eye-to-Hand**), offline session replay, tripod drift monitoring, and USB link diagnostics for robotic vision setups.

## Features

- **Pluggable Camera Backends**: Supports `realsense`, `orbbec`, `uvc`, and offline `sim` backends with subprocess SDK isolation and automatic single-device discovery (`serial: ""`).
- **Eye-in-Hand Calibration**: Evaluates 7 OpenCV hand-eye solvers with IPPE planar pose disambiguation, joint nonlinear Bundle Adjustment, Flange/TCP decoupling, and tabletop/depth plane consistency checks.
- **Eye-to-Hand Calibration**: Chains a wrist-mounted camera with a fixed RGB-D camera over a shared tabletop ArUco marker, enforcing hold-out pose verification and depth scale estimation before publishing extrinsics.
- **Offline Session Replay**: Recomputes recorded sessions and updates Eye-in-Hand tool TCP offsets (`--tcp-offset`) without importing hardware SDKs or Qt.
- **Diagnostics**: Scene-locked tripod displacement detection (`OK` / `WARN` / `FAIL`) and Linux `sysfs` USB 3.x link speed & frame-drop stability testing.
- **Desktop GUI & Headless CLI**: Five-page PySide6 dark desktop interface alongside scriptable, non-GUI CLI subcommands protected by cross-process device leases.

## Project Structure

```text
src/camera_tool/
├── core/           # SE(3) math, tool safety checks, camera distortion models, ArUco gates, session I/O
├── devices/        # Generic backends (realsense, orbbec, uvc, sim), process isolation, robot controllers
├── calibration/    # Eye-in-Hand and Eye-to-Hand solvers, waypoint planners, and workflows
├── diagnostics/    # Tripod scene-lock monitoring and Linux sysfs USB link inspection
├── maintenance/    # Wrist intrinsics export, pointer corner calibration, and extrinsics verification
├── services/       # Qt-free task orchestration and cross-process device lease manager
├── ui/             # Five-page PySide6 desktop interface and 3D OpenGL scene viewer
└── resources/      # Parameterized YAML templates, udev helper script, and native C++ bridge source
```

## Installation

Requires Python `>= 3.12` on Linux.

```bash
# Core package + GUI dependencies
pip install -e .

# Optional hardware SDKs and test runner
pip install -e ".[realsense,orbbec,robot,dev]"
```

- Keep `opencv-python` installed for GUI/OpenCV window support; do not install `opencv-python-headless` in the same environment.
- For Orbbec USB permissions on Linux, install udev rules if needed:
  ```bash
  sudo bash src/camera_tool/resources/install_orbbec_udev_rules.sh
  ```

## Usage

### 1. Desktop Interface

```bash
camera-tool          # Launch 5-page desktop GUI
camera-tool --sim    # Launch desktop GUI in offline simulation mode
```

### 2. Camera Discovery & Preview

```bash
camera-tool preview --list
camera-tool preview --backend realsense
camera-tool preview --backend orbbec --width 1280 --height 800 --fps 30
camera-tool preview --backend sim --headless
```

### 3. Hand-Eye Calibration

Robot controller IP can be provided via `--robot-ip`, the `XARM_IP` environment variable, or a custom YAML file.

```bash
# Offline simulated Eye-in-Hand run
camera-tool calibrate --mode eye-in-hand --sim --result result/eye_in_hand.yaml

# Read-only hardware check (verifies streams, ArUco quality, and robot state without motion)
camera-tool calibrate --mode eye-in-hand --preset default --robot-ip <ROBOT_IP> --check

# Full automatic Eye-in-Hand calibration (presets: default, medium_range, orbbec)
camera-tool calibrate --mode eye-in-hand --preset medium_range --robot-ip <ROBOT_IP>

# Full Eye-to-Hand calibration
camera-tool calibrate --mode eye-to-hand --robot-ip <ROBOT_IP>
```

### 4. Offline Replay & TCP Offset Update

```bash
# Recompute a recorded Eye-in-Hand or Eye-to-Hand session offline
camera-tool replay runs/eye_in_hand/<SESSION_DIR>

# Re-export Eye-in-Hand extrinsics with a new tool TCP offset (x,y,z,r,p,y in mm,rad)
camera-tool replay runs/eye_in_hand/<SESSION_DIR> \
  --tcp-offset 0,0,172,0,0,0 \
  --result result/eye_in_hand_tcp.yaml
```

### 5. Tripod Monitoring & USB Stability Test

```bash
# Lock stationary scene baseline from published Eye-to-Hand extrinsics
camera-tool tripod lock --result result/eye_to_hand.yaml

# Single-frame drift check (returns exit code 2 on confirmed FAIL)
camera-tool tripod watch --once

# Headless USB link speed and frame-gap stability test
camera-tool usb-test --backend realsense --duration 1800 --interval 5 --csv usb_stability.csv
```

## Configuration & Data Conventions

- **Templates**: Built-in YAML templates live under `src/camera_tool/resources/configs/`:
  - `eye_in_hand/default.yaml` — Close-range RGB-D preset (`waypoints/close_range.yaml`).
  - `eye_in_hand/medium_range.yaml` — Medium-range flange bracket preset (`waypoints/medium_range.yaml`).
  - `eye_in_hand/orbbec.yaml` — `1280x800@30` Orbbec RGB-D preset.
  - `eye_in_hand/waypoints/profiles.yaml` — Kinematic workspace profiles (`arm_6dof`, `arm_5dof`, `arm_7dof`, `arm_compact_6dof`, `arm_long_6dof`).
  - `eye_to_hand.yaml` & `corners.yaml` — Eye-to-Hand and TCP pointer corner calibration templates.
- **Hardware Neutrality**: All tracked templates leave `serial: ""` and `robot.ip: ""`. Single-device setups auto-select the connected camera; multi-camera setups can pass `--serial <SN>` or override YAML configs locally.
- **Local Runtime Directories**: Session recordings (`runs/`), published extrinsics (`result/`), and device-specific wrist intrinsics (`intrinsics/wrist_<SERIAL>.json`) are stored locally and excluded from version control.

## Testing

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```
