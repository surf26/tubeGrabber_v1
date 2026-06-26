#!/usr/bin/env python3
"""
test_arm.py — 机械臂驱动最小测试（不依赖相机 / 标定 / YOLO）

用法：
    # 只读当前位姿（默认，最安全）
    python test_arm.py

    # 在当前位姿基础上 z 抬高 3cm 试动
    python test_arm.py --move --dz 0.03

    # 测试夹爪开合（需先 init）
    python test_arm.py --gripper
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from config import CFG
from arm import ArmController


def _fmt_pose(pose: list[float] | None) -> str:
    if pose is None or len(pose) < 6:
        return "<无法读取>"
    x, y, z, rx, ry, rz = pose[:6]
    return (
        f"x={x:+.4f} y={y:+.4f} z={z:+.4f} m  |  "
        f"rx={rx:+.4f} ry={ry:+.4f} rz={rz:+.4f} rad  "
        f"(rx={math.degrees(rx):.1f}°)"
    )


def _pose_delta(a: list[float], b: list[float]) -> float:
  """两点位姿之间的位置距离 (m)。"""
  return float(np.linalg.norm(np.array(b[:3]) - np.array(a[:3])))


def main() -> int:
    ap = argparse.ArgumentParser(description="机械臂驱动最小测试")
    ap.add_argument("--move", action="store_true",
                    help="执行小幅 z 方向试动（默认只读位姿）")
    ap.add_argument("--dz", type=float, default=0.03,
                    help="试动时 z 方向偏移量，米（默认 0.03 = 3cm）")
    ap.add_argument("--speed", type=int, default=10,
                    help="运动速度 %%（默认 10，较慢更安全）")
    ap.add_argument("--gripper", action="store_true",
                    help="测试夹爪 init + 开闭（会动夹爪）")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="跳过运动前确认")
    args = ap.parse_args()

    ac = CFG["arm"]
    print("=" * 60)
    print("  test_arm — 机械臂驱动测试")
    print("=" * 60)
    print(f"  目标: {ac['ip']}:{ac['port']}")
    print(f"  模式: {'试动' if args.move else '只读'}", end="")
    if args.gripper:
        print(" + 夹爪", end="")
    print("\n")

    arm = ArmController()

    try:
        pose0 = arm.get_pose6d()
        print(f"[1] 当前位姿: {_fmt_pose(pose0)}")
        if pose0 is None:
            print("\n[FAIL] 读不到末端位姿。")
            print("  检查: IP/端口、机械臂是否上电使能、有无报警、是否被示教器占用。")
            return 1

        if args.move:
            target = [float(v) for v in pose0[:6]]
            target[2] += args.dz

            print(f"\n[2] 计划 movej_p:")
            print(f"    当前 z = {pose0[2]:+.4f} m")
            print(f"    目标 z = {target[2]:+.4f} m  (dz={args.dz:+.3f} m)")
            print(f"    完整目标 = {[round(v, 4) for v in target]}")
            print(f"    速度 = {args.speed}%")

            if not args.yes:
                ans = input("\n确认执行试动? [y/N] ").strip().lower()
                if ans not in ("y", "yes"):
                    print("已取消。")
                    return 0

            print("\n[3] 发送 movej_p ...")
            ok = arm.movej_p(target, speed=args.speed)
            pose1 = arm.get_pose6d()
            print(f"    返回: {'成功' if ok else '失败'}")
            print(f"    运动后位姿: {_fmt_pose(pose1)}")

            if pose1 is not None:
                dist = _pose_delta(pose0, pose1)
                print(f"    实际位移: {dist * 1000:.1f} mm")
                if ok and dist < 0.001:
                    print("\n[WARN] SDK 返回成功但几乎没动。")
                    print("  常见原因: 未使能、有报警、示教模式、或目标与当前位姿相同。")
                elif not ok:
                    print("\n[FAIL] movej_p 失败，请看上方 code。")
                    print("  常见原因: 目标超出工作空间、姿态不可达 (rx=π 等)。")
                    return 1
                else:
                    print("\n[OK] 试动完成，驱动链路正常。")

            # 可选：回到原位
            if args.yes or input("\n是否回到原位? [y/N] ").strip().lower() in ("y", "yes"):
                print("[4] 回到原位 ...")
                arm.movej_p([float(v) for v in pose0[:6]], speed=args.speed)

        if args.gripper:
            print("\n[夹爪] 初始化（约 5s）...")
            arm.init_gripper()
            gr = CFG["gripper"]
            print(f"[夹爪] 开爪 openness={gr['open_val']}")
            arm.set_gripper(gr["open_val"])
            print(f"[夹爪] 闭爪 openness={gr['close_val']}")
            arm.set_gripper(gr["close_val"])
            print("[OK] 夹爪测试完成。")

        if not args.move and not args.gripper:
            print("\n[OK] 只读测试通过。要试动请加:  python test_arm.py --move")
        return 0

    finally:
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(main())
