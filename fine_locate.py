"""
fine_locate.py — Phase 3 视觉部分：Hand 相机 OpenCV 精定位

用 hand 相机（末端）采多帧，
通过 HSV 阈值 + 轮廓分析找到试管盖的精确像素中心。
"""

import math
import time
from typing import Optional

import cv2
import numpy as np

from config import CFG


def find_cap_center(color: np.ndarray) -> Optional[tuple]:
    """
    单帧图像中找试管盖中心像素 (u, v)。

    步骤：
      1. BGR → HSV，inRange 提取盖颜色
      2. 形态学开运算 + 闭运算去噪
      3. 找轮廓，按圆形度过滤
      4. fitEllipse 精确中心（失败则用质心）
    """
    hv = CFG["hand_vision"]
    lo = np.array(hv["hsv_lower"], dtype=np.uint8)
    hi = np.array(hv["hsv_upper"], dtype=np.uint8)
    ks = hv["morph_kernel"]
    min_circ = hv["min_circularity"]
    min_area = hv["min_area_px"]

    hsv  = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 找最圆的轮廓
    best_cnt  = None
    best_circ = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6 or len(cnt) < 5:
            continue
        circ = 4 * math.pi * area / (peri ** 2)
        if circ >= min_circ and circ > best_circ:
            best_circ = circ
            best_cnt  = cnt

    if best_cnt is None:
        return None

    # 精确中心：优先用椭圆拟合，失败则用质心
    try:
        ellipse    = cv2.fitEllipse(best_cnt)
        cx, cy     = ellipse[0]
    except cv2.error:
        M = cv2.moments(best_cnt)
        if M["m00"] < 1e-6:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    return (float(cx), float(cy))


def get_stable_center(hand_cam) -> tuple:
    """
    采 n_frames 帧，用中位数得到稳定的像素中心 (u, v)。

    Returns: (u_median, v_median, last_frame)
    Raises: RuntimeError 如果有效帧不足
    """
    n = CFG["hand_vision"]["n_frames"]
    print(f"\n[Phase3] ══════ Hand 相机精定位（{n} 帧融合）══════")

    centers   = []
    last_frame = None

    for i in range(n):
        fp = hand_cam.grab()
        if fp is None:
            time.sleep(0.05)
            continue
        last_frame = fp
        center = find_cap_center(fp.color)
        if center is not None:
            centers.append(center)
            print(f"  帧 {i+1:2d}/{n}: 中心=({center[0]:.1f}, {center[1]:.1f})")
        else:
            print(f"  帧 {i+1:2d}/{n}: 未检测到目标")
        time.sleep(0.033)

    if len(centers) < max(2, n // 2):
        raise RuntimeError(
            f"[Phase3] 精定位失败：有效帧 {len(centers)}/{n} 不足\n"
            "  → 检查 config.yaml hand_vision.hsv_lower/upper 是否匹配盖颜色"
        )

    arr = np.array(centers)
    u_med = float(np.median(arr[:, 0]))
    v_med = float(np.median(arr[:, 1]))
    print(f"\n  中位数: ({u_med:.2f}, {v_med:.2f})  "
          f"σ_x={np.std(arr[:, 0]):.2f}px  σ_y={np.std(arr[:, 1]):.2f}px  "
          f"有效帧={len(centers)}")

    return u_med, v_med, last_frame
