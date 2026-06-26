"""
config.py — 配置加载

所有其他模块都 import 这里的 CFG 和 ROOT。
"""

import json
from pathlib import Path

import numpy as np
import yaml

# 本项目根目录（tubeGrabber_v1/）
ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    """读取 config.yaml，返回配置字典。"""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_calib_matrix(path: Path) -> tuple[np.ndarray, str]:
    """
    从手眼标定文件加载 4x4 变换矩阵。

    JSON 格式示例::
        {
          "mode": "eye_to_hand",
          "method": "tsai",
          "num_samples": 20,
          "matrix_name": "T_cam2base",
          "matrix": [[...], [...], [...], [...]]
        }

    仍兼容遗留的 .npy 文件。
  """
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "matrix" not in data:
            raise KeyError(f"JSON 缺少 'matrix' 字段: {path}")
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        parts = []
        for key in ("mode", "method", "matrix_name"):
            if data.get(key):
                parts.append(f"{key}={data[key]!r}")
        if data.get("num_samples") is not None:
            parts.append(f"num_samples={data['num_samples']}")
        info = ", ".join(parts)
    elif suffix == ".npy":
        matrix = np.load(str(path))
        info = "legacy .npy"
    else:
        raise ValueError(f"不支持的标定文件格式: {path.suffix}（支持 .json / .npy）")

    if matrix.shape != (4, 4):
        raise ValueError(f"标定矩阵 shape 应为 (4,4)，实际为 {matrix.shape}: {path}")
    return matrix, info


# 全局配置对象，其他模块直接 from config import CFG, ROOT
CFG = load_config()
