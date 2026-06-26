"""
main.py — 三阶段试管盖抓取主流程

Phase 1：Head 相机 YOLO 全局扫描，找出所有试管盖
Phase 2：ETH 矩阵解算 3D 坐标，机械臂粗移到目标上方
Phase 3：Hand 相机 OpenCV 精定位，EIH 矩阵精确夹取

运行方式：
    cd tubeGrabber_v1
    python main.py
"""

import math
import sys
import time

import numpy as np

from config import CFG, ROOT
from camera import CameraDriver
from arm import ArmController
from detect import scan_tubes
from fine_locate import get_stable_center
from transforms import pixel_to_camera, cam_to_base_eth, cam_to_base_eih, depth_at_point


# =============================================================================
# Phase 2：ETH 解算 + 粗移到目标上方
# =============================================================================

def phase2_move_above(tube: dict, T_cam2base: np.ndarray, arm: ArmController) -> np.ndarray:
    """
    用 ETH 矩阵将 head 像素坐标解算为基座系 3D 坐标，
    PTP 移到目标上方（rx=π 夹爪竖直朝下）。

    Returns: p_base 粗坐标 (3,)
    """
    fp   = tube["frame"]
    u, v = tube["pixel"]
    z    = tube["depth_m"]

    p_cam  = pixel_to_camera(u, v, z, fp.K, fp.dist)
    p_base = cam_to_base_eth(p_cam, T_cam2base)

    print(f"\n[Phase2] ══════ ETH 3D 解算 ══════")
    print(f"  P_cam  = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
    print(f"  P_base = ({p_base[0]:+.4f}, {p_base[1]:+.4f}, {p_base[2]:+.4f}) m")

    g        = CFG["grasp"]
    hover_z  = g["hover_z"]
    target   = [float(p_base[0]), float(p_base[1]), float(p_base[2]) + hover_z,
                math.pi, 0.0, 0.0]   # rx=π → 夹爪竖直朝下

    print(f"\n[Phase2] ══════ 粗移到上方 ══════")
    print(f"  目标: x={target[0]:+.4f}  y={target[1]:+.4f}  z={target[2]:+.4f}  "
          f"(悬停高度 +{hover_z} m)")

    if not arm.movej_p(target, speed=g["speed_coarse"]):
        raise RuntimeError("[Phase2] movej_p 失败")

    print(f"[Phase2] ✓ 到位，等待 {g['settle_s']}s 振动衰减...")
    time.sleep(g["settle_s"])
    return p_base


# =============================================================================
# Phase 3：EIH 精定位 + 夹取
# =============================================================================

def phase3_fine_grasp(p_base_coarse: np.ndarray,
                      T_cam_end: np.ndarray,
                      arm: ArmController,
                      hand_cam) -> None:
    """
    1. OpenCV 多帧精定位 → 中位数像素中心
    2. EIH 解算精确 3D 坐标
    3. 精调到 pre 位 → 开爪 → 下压 → 闭爪 → 提起
    """
    g  = CFG["grasp"]
    gr = CFG["gripper"]
    hv = CFG["hand_vision"]

    # ── 精定位：多帧中位数 ────────────────────────────────────────────────────
    u_med, v_med, fp = get_stable_center(hand_cam)

    # ── 读深度 + FK ───────────────────────────────────────────────────────────
    if fp is None:
        raise RuntimeError("[Phase3] hand 相机无帧")

    T_base_end = arm.get_T_base_end()
    if T_base_end is None:
        raise RuntimeError("[Phase3] 无法读取末端 FK")

    z = depth_at_point(fp.depth, u_med, v_med, hv["depth_roi_radius"])
    if z is None:
        raise RuntimeError(f"[Phase3] 深度无效 (u={u_med:.1f}, v={v_med:.1f})")

    # ── EIH 解算 ──────────────────────────────────────────────────────────────
    p_cam  = pixel_to_camera(u_med, v_med, z, fp.K, fp.dist)
    p_fine = cam_to_base_eih(p_cam, T_cam_end, T_base_end)

    print(f"\n[Phase3] ══════ EIH 3D 解算 ══════")
    print(f"  P_cam    = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
    print(f"  P_coarse = ({p_base_coarse[0]:+.4f}, {p_base_coarse[1]:+.4f}, {p_base_coarse[2]:+.4f}) m")
    print(f"  P_fine   = ({p_fine[0]:+.4f}, {p_fine[1]:+.4f}, {p_fine[2]:+.4f}) m")
    drift = np.linalg.norm(p_fine - p_base_coarse)
    print(f"  粗→精偏移: {drift * 1000:.1f} mm")

    xf, yf, zf = float(p_fine[0]), float(p_fine[1]), float(p_fine[2])

    # ── Step A: 精调到 pre 位 ─────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ 精调 → pre 位 ══════")
    pre_pose = [xf, yf, zf + g["pre_z"], math.pi, 0.0, 0.0]
    if not arm.movej_p(pre_pose, speed=g["speed_fine"]):
        raise RuntimeError("[Phase3] 精调 movej_p 失败")
    time.sleep(g["settle_s"])

    # ── Step B: 开爪 ──────────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ 开爪 (openness={gr['open_val']}) ══════")
    arm.set_gripper(gr["open_val"])
    time.sleep(0.3)

    # ── Step C: 竖直下压 ──────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ ↓ 下压到接触位 ══════")
    touch_pose = [xf, yf, zf + g["touch_z"], math.pi, 0.0, 0.0]
    if not arm.movel(touch_pose, speed=g["speed_grasp"]):
        raise RuntimeError("[Phase3] movel 下压失败")

    # ── Step D: 闭爪 ──────────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ 闭爪 (openness={gr['close_val']}) ══════")
    arm.set_gripper(gr["close_val"])
    time.sleep(0.5)

    # ── Step E: 竖直提起 ──────────────────────────────────────────────────────
    print(f"\n[Phase3] ══════ ↑ 竖直提起 ══════")
    lift_pose = [xf, yf, zf + g["lift_z"], math.pi, 0.0, 0.0]
    if not arm.movel(lift_pose, speed=g["speed_lift"]):
        raise RuntimeError("[Phase3] movel 提起失败")

    print("\n[Phase3] ✓ 夹取完成！")


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("=" * 60)
    print("   tubeGrabber_v1  三阶段试管盖抓取")
    print("=" * 60)

    # ── 加载手眼标定矩阵 ──────────────────────────────────────────────────────
    calib = CFG["calib"]
    p_eth = (ROOT / calib["T_camera_to_base"]).resolve()
    p_eih = (ROOT / calib["T_cam_end"]).resolve()

    for path, name in [(p_eth, "ETH T_camera_to_base"), (p_eih, "EIH T_cam_end")]:
        if not path.exists():
            print(f"[ERROR] 找不到标定矩阵 [{name}]: {path}")
            print("  → 将 .npy 文件放到 tubeGrabber_v1/assets/calib/")
            sys.exit(1)

    T_cam2base = np.load(str(p_eth))
    T_cam_end  = np.load(str(p_eih))
    print(f"[Init] ✓ ETH: {p_eth.name}  shape={T_cam2base.shape}")
    print(f"[Init] ✓ EIH: {p_eih.name}  shape={T_cam_end.shape}")

    # ── 连接机械臂 ────────────────────────────────────────────────────────────
    arm = ArmController()

    # 首次使用需初始化夹爪，取消下行注释（约 5 秒）：
    # arm.init_gripper()

    # ── 开启相机 ──────────────────────────────────────────────────────────────
    cc       = CFG["cameras"]
    head_cam = CameraDriver("head", cc["head_serial"], cc["width"], cc["height"])
    hand_cam = CameraDriver("hand", cc["hand_serial"], cc["width"], cc["height"])

    print("\n[Init] 开启 head 相机...")
    if not head_cam.open():
        print("[ERROR] head 相机开启失败")
        arm.disconnect()
        sys.exit(1)

    print("[Init] 开启 hand 相机...")
    if not hand_cam.open():
        print("[ERROR] hand 相机开启失败")
        head_cam.close()
        arm.disconnect()
        sys.exit(1)

    print("[Init] ✓ 两路相机就绪\n")

    try:
        # ── Phase 1：YOLO 扫描 ────────────────────────────────────────────────
        tube_map = scan_tubes(head_cam)

        if not tube_map:
            print("\n[ERROR] 未检测到试管盖")
            print("  → 检查 YOLO 模型路径 / conf 阈值 / head 相机视野")
            return

        # ── 用户选择目标 ──────────────────────────────────────────────────────
        print(f"\n请输入要抓取的试管编号 [1 ~ {len(tube_map)}]: ", end="", flush=True)
        while True:
            try:
                tid = int(input().strip())
                if tid in tube_map:
                    break
                print(f"  无效编号，请重新输入 [1 ~ {len(tube_map)}]: ", end="", flush=True)
            except ValueError:
                print("  请输入整数: ", end="", flush=True)

        sel = tube_map[tid]
        u, v = sel["pixel"]
        print(f"\n[选择] 试管 [{tid}]  像素=({u:.1f}, {v:.1f})"
              f"  z={sel['depth_m']:.3f} m  conf={sel['confidence']:.3f}")

        # ── Phase 2：粗移到上方 ───────────────────────────────────────────────
        p_coarse = phase2_move_above(sel, T_cam2base, arm)

        # ── Phase 3：精定位 + 夹取 ────────────────────────────────────────────
        phase3_fine_grasp(p_coarse, T_cam_end, arm, hand_cam)

        print("\n" + "=" * 60)
        print("   ✓  全流程完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[用户中断]")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        print("\n[Cleanup] 释放资源...")
        head_cam.close()
        hand_cam.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
