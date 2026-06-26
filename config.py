"""
config.py — 配置加载 + RM_API2 SDK 路径配置

所有其他模块都 import 这里的 CFG 和 ROOT。
"""

import os
import sys
import platform
from pathlib import Path

import yaml

# 本项目根目录（tubeGrabber_v1/）
ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    """读取 config.yaml，返回配置字典。"""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_rm_sdk():
    """
    按优先级查找 RM_API2 SDK 并配置 sys.path + LD_LIBRARY_PATH：
      1. 环境变量 TUBE_RM_API2
      2. tubeGrabber_v1/third_party/RM_API2
    """
    candidates = []
    if os.environ.get("TUBE_RM_API2"):
        candidates.append(Path(os.environ["TUBE_RM_API2"]))
    candidates.append(ROOT / "third_party" / "RM_API2")

    for rm_root in candidates:
        py_dir = rm_root / "Python"
        if not (py_dir / "Robotic_Arm").is_dir():
            continue

        # 加入 Python 搜索路径
        if str(py_dir) not in sys.path:
            sys.path.insert(0, str(py_dir))

        # 加入 native 库路径
        arch = "linux_aarch64" if "aarch64" in platform.machine() else "linux_x86"
        lib_dir = str(py_dir / "Robotic_Arm" / "libs" / arch)
        if os.path.isdir(lib_dir):
            old_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in old_ld:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + old_ld if old_ld else "")

        print(f"[SDK] RM_API2 已找到: {rm_root}")
        return

    raise RuntimeError(
        "找不到 RM_API2 SDK！\n"
        "  方法1: export TUBE_RM_API2=/path/to/RM_API2\n"
        "  方法2: 将 RM_API2 放到 tubeGrabber_v1/third_party/RM_API2"
    )


# 全局配置对象，其他模块直接 from config import CFG, ROOT
CFG = load_config()
