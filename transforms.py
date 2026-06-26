"""
transforms.py — 坐标变换工具

包含：
  - 像素 + 深度 → 相机系 3D 坐标
  - ETH（head 固定相机）变换到基座系
  - EIH（hand 末端相机）变换到基座系
  - 从深度图采样深度值（bbox 中位数 / 点邻域中位数）
"""

from typing import Optional

import cv2
import numpy as np

from config import CFG

DEPTH_MIN: float = CFG["depth"]["min_m"]
DEPTH_MAX: float = CFG["depth"]["max_m"]


# =============================================================================
# 像素 → 3D
# =============================================================================

def pixel_to_camera(u: float, v: float, z: float,
                    K: np.ndarray,
                    dist: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    像素坐标 (u, v) + 深度 z（米）→ 相机系坐标 [xc, yc, zc]。

    若有畸变系数 dist，先去畸变再反投影。
    """
    if z <= 0:
        return None

    # 去畸变
    if dist is not None and np.any(dist != 0):
        pt  = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, K, dist, P=K)[0, 0]
        u, v = float(und[0]), float(und[1])

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx,
                     (v - cy) * z / fy,
                     z], dtype=np.float64)


# =============================================================================
# 相机系 → 基座系
# =============================================================================

def cam_to_base_eth(p_cam: np.ndarray, T_cam2base: np.ndarray) -> np.ndarray:
    """
    ETH（head 固定相机）：
        P_base = T_camera_to_base @ [p_cam, 1]^T
    """
    ph = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T_cam2base @ ph)[:3]


def cam_to_base_eih(p_cam: np.ndarray,
                    T_cam_end: np.ndarray,
                    T_base_end: np.ndarray) -> np.ndarray:
    """
    EIH（hand 末端相机）：
        P_base = T_base_end @ T_cam_end @ [p_cam, 1]^T
    """
    # 构造齐次点的平移矩阵
    T_ct = np.eye(4)
    T_ct[:3, 3] = p_cam
    return (T_base_end @ T_cam_end @ T_ct)[:3, 3]


# =============================================================================
# 深度采样
# =============================================================================

def depth_at_bbox(depth: np.ndarray, bbox: list, shrink: float = 0.10) -> Optional[float]:
    """
    在 YOLO bbox 内部（向内缩 shrink 比例）取深度中位数（米）。
    有效深度范围：DEPTH_MIN ~ DEPTH_MAX。
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    # 向内缩，避免 bbox 边缘深度噪声
    mx  = max(1, int((x2 - x1) * shrink))
    my  = max(1, int((y2 - y1) * shrink))
    roi = depth[y1 + my: y2 - my, x1 + mx: x2 - mx]
    if roi.size == 0:
        return None

    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) >= 10 else None


def depth_at_point(depth: np.ndarray, u: float, v: float, radius: int = 14) -> Optional[float]:
    """
    在像素 (u, v) 的圆形邻域内取深度中位数（米）。
    """
    H, W = depth.shape
    uu, vv = int(round(u)), int(round(v))
    roi = depth[max(0, vv - radius): min(H, vv + radius + 1),
                max(0, uu - radius): min(W, uu + radius + 1)]
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) >= 5 else None
