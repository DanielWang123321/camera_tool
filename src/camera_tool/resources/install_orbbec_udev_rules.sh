#!/usr/bin/env bash
set -euo pipefail

RULE_FILE="/etc/udev/rules.d/99-obsensor-libusb.rules"

if [[ "${EUID}" -ne 0 ]]; then
    echo "请使用 sudo 运行此脚本：sudo bash $0" >&2
    exit 1
fi

cat > "${RULE_FILE}" <<'EOF'
# Orbbec USB 3D cameras (idVendor=2bc5) libusb & hidraw access
SUBSYSTEM=="usb", ATTRS{idVendor}=="2bc5", MODE="0666", TAG+="uaccess"
KERNEL=="hidraw*", ATTRS{idVendor}=="2bc5", MODE="0666", TAG+="uaccess"
EOF

udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=2bc5
udevadm trigger --subsystem-match=hidraw
echo "已安装并生效：${RULE_FILE}"

