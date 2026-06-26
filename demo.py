#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tubeGrabber_v1/demo.py  ── 三阶段试管盖抓取 Demo（完全独立版）
================================================================
本文件不依赖任何外部项目目录，所有代码自洽于 tubeGrabber_v1/。

目录结构要求：
    tubeGrabber_v1/
    ├── demo.py             ← 本文件
    ├── camera.py           ← Orbbec 相机驱动
    ├── config.yaml         ← 配置
    ├── assets/
    │   ├── model/best.pt   ← YOLO 模型（自行放置）
    │   └── calib/
    │       ├── T_camera_to_base.npy   ← ETH 标定矩阵
    │       └── T_cam_end.npy          ← EIH 标定矩阵
    └── third_party/
        └── RM_API2/        ← Realman SDK（或设 TUBE_RM_API2 环境变量）

运行：
    cd /path/to/tubeGrabber_v1
    python demo.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── 路径：所有路径只相对本文件所在目录 ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent   # tubeGrabber_v1/

# ── 将本目录加入 sys.path（让 camera.py 可以 import）─────────────────────────
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 查找并配置 RM_API2 SDK ────────────────────────────────────────────────────
def _setup_rm_sdk():
    """
    按以下优先级查找 RM_API2 并配置 sys.path + LD_LIBRARY_PATH：
        1. 环境变量 TUBE_RM_API2
        2. tubeGrabber_v1/third_party/RM_API2
    """
    candidates: List[Path] = []
    if os.environ.get("TUBE_RM_API2"):
        candidates.append(Path(os.environ["TUBE_RM_API2"]))
    candidates.append(_ROOT / "third_party" / "RM_API2")

    for rm_root in candidates:
        py_dir = rm_root / "Python"
        if not (py_dir / "Robotic_Arm").is_dir():
            continue
        # 加入 Python path
        if str(py_dir) not in sys.path:
            sys.path.insert(0, str(py_dir))
        # 加入 native lib
        import platform
        arch    = "linux_aarch64" if "aarch64" in platform.machine().lower() else "linux_x86"
        lib_dir = str(py_dir / "Robotic_Arm" / "libs" / arch)
        if os.path.isdir(lib_dir):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in ld:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")
        print(f"[SDK] RM_API2 已找到: {rm_root}")
        return True

    raise RuntimeError(
        "找不到 RM_API2 SDK！\n"
        "  方法1: export TUBE_RM_API2=/path/to/RM_API2\n"
        "  方法2: 将 RM_API2 放到 tubeGrabber_v1/third_party/RM_API2"
    )


# ── 配置加载 ──────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    p = _ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = _load_config()
DEPTH_MIN: float = CFG["depth"]["min_m"]
DEPTH_MAX: float = CFG["depth"]["max_m"]


# =============================================================================
# 坐标变换工具函数
# =============================================================================

def pixel_to_camera(u: float, v: float, z: float,
                    K: np.ndarray,
                    dist: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """像素 (u,v) + 深度 z(m) → 相机系坐标 [xc,yc,zc]（针孔反投影）。"""
    if z <= 0:
        return None
    if dist is not None and np.any(dist != 0):
        pt  = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, K, dist, P=K)[0, 0]
        u, v = float(und[0]), float(und[1])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx,
                     (v - cy) * z / fy,
                     z], dtype=np.float64)


def cam_to_base_eth(p_cam: np.ndarray, T_cam2base: np.ndarray) -> np.ndarray:
    """ETH（head 固定相机）: P_base = T_camera_to_base @ [p_cam,1]^T"""
    ph = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T_cam2base @ ph)[:3]


def cam_to_base_eih(p_cam: np.ndarray,
                    T_cam_end:  np.ndarray,
                    T_base_end: np.ndarray) -> np.ndarray:
    """EIH（hand 末端相机）: P_base = T_base_end @ T_cam_end @ [p_cam,1]^T"""
    T_ct = np.eye(4)
    T_ct[:3, 3] = p_cam
    return (T_base_end @ T_cam_end @ T_ct)[:3, 3]


def depth_at_bbox(depth: np.ndarray, bbox: List[float], shrink: float = 0.10) -> Optional[float]:
    """YOLO bbox 内缩 shrink 后取深度中位数（米）。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mx  = max(1, int((x2 - x1) * shrink))
    my  = max(1, int((y2 - y1) * shrink))
    roi = depth[y1 + my: y2 - my, x1 + mx: x2 - mx]
    if roi.size == 0:
        return None
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) >= 10 else None


def depth_at_point(depth: np.ndarray, u: float, v: float, r: int = 14) -> Optional[float]:
    """(u,v) 圆形邻域取深度中位数（米）。"""
    H, W = depth.shape
    uu, vv = int(round(u)), int(round(v))
    roi   = depth[max(0, vv - r): min(H, vv + r + 1),
                  max(0, uu - r): min(W, uu + r + 1)]
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) >= 5 else None


# =============================================================================
# 机械臂控制器
# =============================================================================

class ArmController:
    """封装 Realman RM-65B SDK（线程安全）。"""

    def __init__(self, ip: str, port: int):
        import threading
        _setup_rm_sdk()
        from Robotic_Arm.rm_robot_interface import (  # type: ignore
            RoboticArm, rm_modbus_rtu_write_params_t, rm_thread_mode_e
        )
        self._lock      = threading.Lock()
        self._RmParams  = rm_modbus_rtu_write_params_t
        self.arm        = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle     = self.arm.rm_create_robot_arm(ip, port)
        self._openness  = 0.0
        self._run_speed = CFG["gripper"]["run_speed"]
        print(f"[Arm] 已连接 {ip}:{port}  handle={self.handle.id}")

    # ── 读取状态 ──────────────────────────────────────────────────────────────

    def get_pose6d(self) -> Optional[List[float]]:
        """当前末端位姿 [x,y,z,rx,ry,rz]（米+弧度）。"""
        with self._lock:
            ret, state = self.arm.rm_get_current_arm_state()
        return list(state.get("pose", [])) if ret == 0 else None

    def get_T_base_end(self) -> Optional[np.ndarray]:
        """末端在基座系下的 4×4 变换矩阵（FK）。"""
        from scipy.spatial.transform import Rotation
        pose = self.get_pose6d()
        if pose is None or len(pose) < 6:
            return None
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("ZYX", pose[3:6], degrees=False).as_matrix()
        T[:3,  3] = pose[:3]
        return T

    # ── 运动指令 ──────────────────────────────────────────────────────────────

    def movej_p(self, pose6d: List[float], speed: int = 25) -> bool:
        """笛卡尔 PTP（rm_movej_p），阻塞直到到位。"""
        with self._lock:
            code = self.arm.rm_movej_p(pose6d, speed, 0, 0, 1)
        if code != 0:
            print(f"[Arm] rm_movej_p 失败 code={code}")
        return code == 0

    def movel(self, pose6d: List[float], speed: int = 15) -> bool:
        """笛卡尔直线（rm_movel），阻塞。竖直段必须用此接口。"""
        with self._lock:
            code = self.arm.rm_movel(pose6d, speed, 0, 0, 1)
        if code != 0:
            print(f"[Arm] rm_movel 失败 code={code}")
        return code == 0

    # ── 夹爪（RS485 Modbus RTU）──────────────────────────────────────────────

    def init_gripper(self):
        """初始化夹爪（~5秒，首次使用前调用一次）。"""
        g = CFG["gripper"]
        self.arm.rm_set_tool_voltage(3);      time.sleep(0.5)
        self.arm.rm_set_tool_rs485_mode(0, 9600); time.sleep(0.2)
        self._wr(36, g["zero_speed"]);        time.sleep(0.2)
        self._wr(38, g["init_speed"]);        time.sleep(0.2)
        self._wr(40, g["run_speed"]);         time.sleep(0.2)
        self._wr(43, 256000)                  # 全闭找零
        time.sleep(5.0)
        self._openness = 0.0
        print("[Arm] 夹爪初始化完成")

    def set_gripper(self, openness: float):
        """openness: 0.0=全闭, 1.0=全开。"""
        openness = float(np.clip(openness, 0.0, 1.0))
        pos      = int((1.0 - openness) * 256000)
        wait_t   = abs(self._openness - openness) * 256000 / self._run_speed
        self._wr(43, pos)
        self._openness = openness
        time.sleep(wait_t + 0.1)

    def _wr(self, address: int, value: int) -> int:
        high  = (value >> 16) & 0xFFFF
        low   = value & 0xFFFF
        param = self._RmParams(device=1, address=address, type=1, num=2,
                               data=[high, low])
        return self.arm.rm_write_modbus_rtu_registers(param)

    def disconnect(self):
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass
        print("[Arm] 已断开")


# =============================================================================
# Phase 1：Head 相机 YOLO 全局扫描
# =============================================================================

def phase1_scan(head_cam) -> Dict[int, dict]:
    """
    YOLO 检测所有试管盖，按置信度降序编号（1,2,3...）。

    Returns: {id: {"bbox", "pixel", "confidence", "depth_m", "frame"}}
    """
    yolo_cfg   = CFG["yolo"]
    model_path = (_ROOT / yolo_cfg["model_path"]).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO 模型不存在: {model_path}\n"
            "  → 将 best.pt 放到 tubeGrabber_v1/assets/model/"
        )

    print(f"\n[Phase1] 加载 YOLO: {model_path.name} ...")
    from ultralytics import YOLO  # type: ignore
    yolo = YOLO(str(model_path))
    yolo.to(yolo_cfg["device"])

    # 等待首帧
    fp = None
    for _ in range(20):
        fp = head_cam.grab()
        if fp is not None:
            break
        time.sleep(0.1)
    if fp is None:
        raise RuntimeError("[Phase1] head 相机无帧")

    results = yolo.predict(
        fp.color,
        conf    = yolo_cfg["conf"],
        iou     = yolo_cfg["iou"],
        imgsz   = yolo_cfg["imgsz"],
        verbose = False,
    )

    dets = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = bbox
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            z = depth_at_bbox(fp.depth, bbox)
            if z is None:
                print(f"  深度无效，跳过 bbox={[round(b,1) for b in bbox]}")
                continue
            dets.append({"bbox": bbox, "pixel": (u, v),
                         "confidence": conf, "depth_m": z, "frame": fp})

    dets.sort(key=lambda d: -d["confidence"])

    # ── 打印表格 ──────────────────────────────────────────────────────────────
    print(f"\n[Phase1] ══════ 检测到 {len(dets)} 个试管盖 ══════")
    print(f"  {'编号':>4}   {'像素中心 (u, v)':^22}   {'深度':>8}   {'置信度':>8}")
    print("  " + "─" * 52)
    tube_map: Dict[int, dict] = {}
    for i, d in enumerate(dets, 1):
        u, v = d["pixel"]
        print(f"  [{i:>2}]    ({u:>7.1f}, {v:>7.1f})       z={d['depth_m']:.3f}m    conf={d['confidence']:.3f}")
        tube_map[i] = d
    return tube_map


# =============================================================================
# Phase 2：ETH 解算 3D + 机械臂粗移到上方
# =============================================================================

def phase2_move_above(tube: dict, T_cam2base: np.ndarray, arm: ArmController) -> np.ndarray:
    """
    用 ETH 矩阵将 head 像素坐标解算为基座系 3D 坐标，
    以竖直向下姿态（rx=π, ry=0, rz=0）PTP 移到目标上方。

    Returns: p_base_coarse (3,)
    """
    fp   = tube["frame"]
    u, v = tube["pixel"]
    z    = tube["depth_m"]

    p_cam  = pixel_to_camera(u, v, z, fp.K, fp.dist)
    if p_cam is None:
        raise RuntimeError("[Phase2] pixel_to_camera 失败")

    p_base = cam_to_base_eth(p_cam, T_cam2base)

    print(f"\n[Phase2] ══════ ETH 3D 解算 ══════")
    print(f"  P_cam   = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
    print(f"  P_base  = ({p_base[0]:+.4f}, {p_base[1]:+.4f}, {p_base[2]:+.4f}) m")

    hover_z    = CFG["grasp"]["hover_z"]
    target_z   = float(p_base[2]) + hover_z
    target_pose = [float(p_base[0]), float(p_base[1]), target_z,
                   math.pi, 0.0, 0.0]    # rx=π 夹爪竖直朝下

    print(f"\n[Phase2] ══════ rm_movej_p → 上方 ══════")
    print(f"  目标: x={target_pose[0]:+.4f} y={target_pose[1]:+.4f} "
          f"z={target_pose[2]:+.4f}  rx=π ry=0 rz=0")
    print(f"  (悬停高度 z_target + {hover_z}m)")

    if not arm.movej_p(target_pose, speed=CFG["grasp"]["speed_coarse"]):
        raise RuntimeError("[Phase2] rm_movej_p 失败")

    settle = CFG["grasp"]["settle_s"]
    print(f"[Phase2] ✓ 到位，等待 {settle}s 振动衰减...")
    time.sleep(settle)
    return p_base


# =============================================================================
# Phase 3：Hand 相机 OpenCV 精定位 + EIH 夹取
# =============================================================================

def _find_center_opencv(color: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    HSV 阈值 → 形态学 OPEN+CLOSE → findContours → 圆形度过滤 → fitEllipse。
    返回椭圆中心 (u,v) 或 None。
    """
    hv  = CFG["hand_vision"]
    lo  = np.array(hv["hsv_lower"], dtype=np.uint8)
    hi  = np.array(hv["hsv_upper"], dtype=np.uint8)
    ks  = hv["morph_kernel"]
    mc  = hv["min_circularity"]
    ma  = hv["min_area_px"]

    hsv    = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, lo, hi)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_cnt, best_c = None, -1.0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < ma:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6 or len(cnt) < 5:
            continue
        c = 4 * math.pi * area / (peri ** 2)
        if c >= mc and c > best_c:
            best_c   = c
            best_cnt = cnt

    if best_cnt is None:
        return None

    try:
        ellipse = cv2.fitEllipse(best_cnt)
        cx, cy  = ellipse[0]
    except cv2.error:
        M = cv2.moments(best_cnt)
        if M["m00"] < 1e-6:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    return (float(cx), float(cy))


def phase3_fine_and_grasp(p_base_coarse: np.ndarray,
                          T_cam_end:     np.ndarray,
                          arm:           ArmController,
                          hand_cam) -> None:
    """
    1. n_frames 帧 OpenCV 轮廓精定位 → 中位数像素中心
    2. EIH 解算基座系精坐标 P_base_fine
    3. rm_movej_p 精调到 pre 位
    4. 开爪 → rm_movel 下压 → 闭爪 → rm_movel 竖直提起
    """
    hv       = CFG["hand_vision"]
    g        = CFG["grasp"]
    gr       = CFG["gripper"]
    n        = hv["n_frames"]
    r_d      = hv["depth_roi_radius"]
    pre_z    = g["pre_z"]
    touch_z  = g["touch_z"]
    lift_z   = g["lift_z"]
    settle   = g["settle_s"]

    print(f"\n[Phase3] ══════ Hand 相机精定位（{n} 帧融合）══════")

    from camera import FramePacket  # 本目录 camera.py
    centers: List[Tuple[float, float]] = []
    last_fp = None

    for i in range(n):
        fp = hand_cam.grab()
        if fp is None:
            time.sleep(0.05)
            continue
        last_fp = fp
        c = _find_center_opencv(fp.color)
        if c is not None:
            centers.append(c)
            print(f"  帧 {i+1:2d}/{n}: 中心=({c[0]:.1f}, {c[1]:.1f})")
        else:
            print(f"  帧 {i+1:2d}/{n}: 未检测到目标")
        time.sleep(0.033)

    if len(centers) < max(2, n // 2):
        raise RuntimeError(
            f"[Phase3] 精定位失败：有效帧 {len(centers)}/{n} 不足\n"
            "  → 检查 config.yaml hand_vision.hsv_lower/upper 是否匹配盖颜色"
        )

    arr   = np.array(centers)
    u_med = float(np.median(arr[:, 0]))
    v_med = float(np.median(arr[:, 1]))
    print(f"\n  中位数: ({u_med:.2f}, {v_med:.2f})  "
          f"σ_x={np.std(arr[:,0]):.2f}px  σ_y={np.std(arr[:,1]):.2f}px  "
          f"有效帧={len(centers)}")

    # ── 读深度 + FK ──────────────────────────────────────────────────────────
    fp = hand_cam.grab() or last_fp
    if fp is None:
        raise RuntimeError("[Phase3] hand 相机无帧")

    T_base_end = arm.get_T_base_end()
    if T_base_end is None:
        raise RuntimeError("[Phase3] 无法读取末端 FK")

    z = depth_at_point(fp.depth, u_med, v_med, r_d)
    if z is None:
        raise RuntimeError(f"[Phase3] 深度无效 (u={u_med:.1f}, v={v_med:.1f})")

    # ── EIH 解算 ─────────────────────────────────────────────────────────────
    p_cam = pixel_to_camera(u_med, v_med, z, fp.K, fp.dist)
    if p_cam is None:
        raise RuntimeError("[Phase3] pixel_to_camera 失败")

    p_fine = cam_to_base_eih(p_cam, T_cam_end, T_base_end)

    print(f"\n[Phase3] ══════ EIH 3D 解算 ══════")
    print(f"  P_cam   = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
    print(f"  P_coarse= ({p_base_coarse[0]:+.4f}, {p_base_coarse[1]:+.4f}, {p_base_coarse[2]:+.4f}) m")
    print(f"  P_fine  = ({p_fine[0]:+.4f}, {p_fine[1]:+.4f}, {p_fine[2]:+.4f}) m")
    drift = np.linalg.norm(p_fine - p_base_coarse)
    print(f"  粗→精偏移: {drift * 1000:.1f} mm")

    xf, yf, zf = float(p_fine[0]), float(p_fine[1]), float(p_fine[2])

    # ── Step A: 精调到 pre 位（z + pre_z）──────────────────────────────────
    print(f"\n[Phase3] ══════ 精调 → pre 位 ══════")
    pre_pose = [xf, yf, zf + pre_z, math.pi, 0.0, 0.0]
    print(f"  pre: z = {zf:.4f} + {pre_z} = {zf + pre_z:.4f} m  speed={g['speed_fine']}%")
    if not arm.movej_p(pre_pose, speed=g["speed_fine"]):
        raise RuntimeError("[Phase3] 精调 movej_p 失败")
    print(f"[Phase3] 等待 {settle}s 振动衰减...")
    time.sleep(settle)

    # ── Step B: 开爪 ────────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ 开爪 ({gr['open_val']}) ══════")
    arm.set_gripper(gr["open_val"])
    time.sleep(0.3)

    # ── Step C: rm_movel 竖直下压 ────────────────────────────────────────────
    touch_pose = [xf, yf, zf + touch_z, math.pi, 0.0, 0.0]
    print(f"\n[Phase3] ══════ ↓ rm_movel 下压 ══════")
    print(f"  touch: z = {zf:.4f} + {touch_z} = {zf + touch_z:.4f} m  speed={g['speed_grasp']}%")
    if not arm.movel(touch_pose, speed=g["speed_grasp"]):
        raise RuntimeError("[Phase3] rm_movel 下压失败")

    # ── Step D: 闭爪 ────────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ 闭爪 ({gr['close_val']}) ══════")
    arm.set_gripper(gr["close_val"])
    time.sleep(0.5)

    # ── Step E: rm_movel 竖直提起 ────────────────────────────────────────────
    lift_pose = [xf, yf, zf + lift_z, math.pi, 0.0, 0.0]
    print(f"\n[Phase3] ══════ ↑ rm_movel 竖直提起 ══════")
    print(f"  lift: z = {zf:.4f} + {lift_z} = {zf + lift_z:.4f} m  speed={g['speed_lift']}%")
    if not arm.movel(lift_pose, speed=g["speed_lift"]):
        raise RuntimeError("[Phase3] rm_movel 提起失败")

    print("\n[Phase3] ✓ 夹取完成！")


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("=" * 62)
    print("   tubeGrabber_v1  三阶段试管盖抓取 Demo（完全独立版）")
    print("=" * 62)

    # ── 加载手眼标定矩阵 ──────────────────────────────────────────────────────
    calib = CFG["calib"]
    p_eth = (_ROOT / calib["T_camera_to_base"]).resolve()
    p_eih = (_ROOT / calib["T_cam_end"]).resolve()

    from config import load_calib_matrix

    for p, name in [(p_eth, "ETH T_cam2base"), (p_eih, "EIH T_cam2end")]:
        if not p.exists():
            print(f"[ERROR] 找不到标定矩阵 [{name}]: {p}")
            print("  → 将 JSON 标定文件放到 tubeGrabber_v1/assets/calib/")
            sys.exit(1)

    try:
        T_cam2base, eth_info = load_calib_matrix(p_eth)
        T_cam_end, eih_info = load_calib_matrix(p_eih)
    except (KeyError, ValueError) as e:
        print(f"[ERROR] 标定矩阵加载失败: {e}")
        sys.exit(1)

    print(f"[Init] ✓ ETH: {p_eth.name}  shape={T_cam2base.shape}  ({eth_info})")
    print(f"[Init] ✓ EIH: {p_eih.name}  shape={T_cam_end.shape}  ({eih_info})")

    # ── 连接机械臂 ────────────────────────────────────────────────────────────
    ac = CFG["arm"]
    print(f"\n[Init] 连接机械臂 {ac['ip']}:{ac['port']} ...")
    arm = ArmController(ac["ip"], ac["port"])

    # 首次使用需要初始化夹爪：取消以下注释
    # print("[Init] 初始化夹爪（约5秒）...")
    # arm.init_gripper()

    # ── 开启相机 ──────────────────────────────────────────────────────────────
    from camera import CameraDriver   # 本目录 camera.py
    cc       = CFG["cameras"]
    head_cam = CameraDriver("head", cc["head_serial"], cc["width"], cc["height"])
    hand_cam = CameraDriver("hand", cc["hand_serial"], cc["width"], cc["height"])

    print("\n[Init] 开启 head 相机...")
    if not head_cam.open():
        print("[ERROR] head 相机开启失败")
        arm.disconnect(); sys.exit(1)

    print("[Init] 开启 hand 相机...")
    if not hand_cam.open():
        print("[ERROR] hand 相机开启失败")
        head_cam.close(); arm.disconnect(); sys.exit(1)

    print("[Init] ✓ 两路相机已就绪\n")

    try:
        # ══════════════════════════════════════════════════════
        # Phase 1：全局扫描 + 用户选择
        # ══════════════════════════════════════════════════════
        tube_map = phase1_scan(head_cam)

        if not tube_map:
            print("\n[ERROR] 未检测到试管盖")
            print("  → 检查 YOLO 模型路径 / conf 阈值 / head 相机视野")
            return

        print(f"\n请输入要抓取的试管编号 [1 ~ {len(tube_map)}]: ", end="", flush=True)
        while True:
            try:
                tid = int(input().strip())
                if tid in tube_map:
                    break
                print(f"  无效编号，请输入 1~{len(tube_map)}: ", end="", flush=True)
            except ValueError:
                print("  请输入整数: ", end="", flush=True)

        sel = tube_map[tid]
        u, v = sel["pixel"]
        print(f"\n[选择] 试管 [{tid}]  像素=({u:.1f},{v:.1f})"
              f"  z={sel['depth_m']:.3f}m  conf={sel['confidence']:.3f}")

        # ══════════════════════════════════════════════════════
        # Phase 2：ETH 3D + 粗移到上方
        # ══════════════════════════════════════════════════════
        p_coarse = phase2_move_above(sel, T_cam2base, arm)

        # ══════════════════════════════════════════════════════
        # Phase 3：Hand 精定位 + 夹取 + 提起
        # ══════════════════════════════════════════════════════
        phase3_fine_and_grasp(p_coarse, T_cam_end, arm, hand_cam)

        print("\n" + "=" * 62)
        print("   ✓  全流程完成！")
        print("=" * 62)

    except KeyboardInterrupt:
        print("\n[用户中断]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Cleanup] 释放资源...")
        head_cam.close()
        hand_cam.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
