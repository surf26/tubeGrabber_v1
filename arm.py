"""
arm.py — Realman RM-65B 机械臂控制器

封装了：
  - 连接 / 断开
  - 读取末端位姿 / FK 变换矩阵
  - PTP 运动（movej_p）、直线运动（movel）
  - 夹爪初始化 + 开闭控制
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
from Robotic_Arm.rm_robot_interface import (
    RoboticArm, rm_modbus_rtu_write_params_t, rm_thread_mode_e
)

from config import CFG


class ArmController:
    """封装 Realman SDK，所有指令线程安全。"""

    def __init__(self):

        ac = CFG["arm"]
        self._lock     = threading.Lock()
        self._RmParams = rm_modbus_rtu_write_params_t
        self._run_speed = CFG["gripper"]["run_speed"]
        self._openness  = 0.0

        self.arm    = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ac["ip"], ac["port"])
        print(f"[Arm] 已连接 {ac['ip']}:{ac['port']}  handle={self.handle.id}")

    # ── 读取状态 ──────────────────────────────────────────────────────────────

    def get_pose6d(self):
        """当前末端位姿 [x, y, z, rx, ry, rz]（米 + 弧度）。"""
        with self._lock:
            ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            return None
        return list(state.get("pose", []))

    def get_T_base_end(self):
        """末端在基座系下的 4×4 变换矩阵（正运动学）。"""
        pose = self.get_pose6d()
        if pose is None or len(pose) < 6:
            return None
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("ZYX", pose[3:6], degrees=False).as_matrix()
        T[:3,  3] = pose[:3]
        return T

    # ── 运动指令 ──────────────────────────────────────────────────────────────

    def movej_p(self, pose6d: list, speed: int = 25) -> bool:
        """笛卡尔 PTP，阻塞到位。pose6d = [x, y, z, rx, ry, rz]"""
        with self._lock:
            code = self.arm.rm_movej_p(pose6d, speed, 0, 0, 1)
        if code != 0:
            print(f"[Arm] movej_p 失败 code={code}")
        return code == 0

    def movel(self, pose6d: list, speed: int = 15) -> bool:
        """笛卡尔直线运动，阻塞到位。竖直段必须用此接口。"""
        with self._lock:
            code = self.arm.rm_movel(pose6d, speed, 0, 0, 1)
        if code != 0:
            print(f"[Arm] movel 失败 code={code}")
        return code == 0

    # ── 夹爪（RS485 Modbus RTU）──────────────────────────────────────────────

    def init_gripper(self):
        """初始化夹爪（约 5 秒，首次使用前调用一次）。"""
        g = CFG["gripper"]
        self.arm.rm_set_tool_voltage(3)
        time.sleep(0.5)
        self.arm.rm_set_tool_rs485_mode(0, 9600)
        time.sleep(0.2)
        self._write_reg(36, g["zero_speed"])
        time.sleep(0.2)
        self._write_reg(38, g["init_speed"])
        time.sleep(0.2)
        self._write_reg(40, g["run_speed"])
        time.sleep(0.2)
        self._write_reg(43, 256000)   # 全闭找零
        time.sleep(5.0)
        self._openness = 0.0
        print("[Arm] 夹爪初始化完成")

    def set_gripper(self, openness: float):
        """
        openness: 0.0 = 全闭，1.0 = 全开。
        自动估算等待时间，确保夹爪到位后再返回。
        """
        openness = float(np.clip(openness, 0.0, 1.0))
        pos      = int((1.0 - openness) * 256000)
        wait_t   = abs(self._openness - openness) * 256000 / self._run_speed
        self._write_reg(43, pos)
        self._openness = openness
        time.sleep(wait_t + 0.1)

    def disconnect(self):
        """释放机械臂连接。"""
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass
        print("[Arm] 已断开")

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _write_reg(self, address: int, value: int) -> int:
        """Modbus RTU 写两个 16 位寄存器（一个 32 位值）。"""
        high  = (value >> 16) & 0xFFFF
        low   = value & 0xFFFF
        param = self._RmParams(device=1, address=address, type=1,
                               num=2, data=[high, low])
        return self.arm.rm_write_modbus_rtu_registers(param)
