# tubeGrabber_v1 — 三阶段试管盖抓取 Demo（完全独立版）

> **本项目完全自洽，不依赖任何外部目录。**  
> 将整个 `tubeGrabber_v1/` 文件夹复制到任意位置均可直接运行。

---

## 文件结构

```
tubeGrabber_v1/
├── demo.py                  ← 主脚本（三阶段完整流程）
├── camera.py                ← Orbbec 相机驱动（内置）
├── config.yaml              ← 所有可调参数
├── assets/
│   ├── model/
│   │   └── best.pt          ← YOLO 模型（需自行放置）
│   └── calib/
│       ├── T_cam2base.json    ← ETH 手眼矩阵（head 固定相机→基座）
│       └── T_cam2end.json     ← EIH 手眼矩阵（hand 相机→末端）
└── README.md
```

---

## 运行前准备

### 1. 依赖安装

```bash
pip install -r requirements.txt
pip install pyorbbecsdk   # Orbbec 相机 SDK
```

### 2. 放置 YOLO 模型

```bash
cp /path/to/best.pt tubeGrabber_v1/assets/model/best.pt
```

### 3. 放置手眼标定矩阵

```bash
cp /path/to/T_cam2base.json tubeGrabber_v1/assets/calib/
cp /path/to/T_cam2end.json  tubeGrabber_v1/assets/calib/
```

> 标定结果为 JSON，需包含 `matrix` 字段（4×4 变换矩阵）。

### 4. 修改 config.yaml

```yaml
arm:
  ip: "192.168.1.18"        # ← 改为机械臂实际 IP

cameras:
  head_serial: "CPCS253000HM"   # ← 改为 head 相机序列号
  hand_serial: "CP84B4100090"   # ← 改为 hand 相机序列号
```

### 5. 首次使用初始化夹爪

取消 `demo.py` 中以下注释（仅首次需要，约5秒）：
```python
# arm.init_gripper()
```

---

## 运行

```bash
cd /path/to/tubeGrabber_v1
python demo.py
```

---

## 三阶段流程详解

### Phase 1：Head 相机 YOLO 全局扫描

```
head 相机采一帧
  → YOLO 推理（best.pt）
  → 每个 bbox 取深度中位数（内缩10%去掉边缘噪声）
  → 按置信度降序编号 [1,2,3,...]
  → 控制台打印编号表格
  → 等待用户输入目标编号
```

### Phase 2：ETH 解算 3D + 粗移到上方

```
选定试管像素中心 (u,v) + 深度 z
  → pixel_to_camera()：针孔反投影 → 相机系坐标 P_cam
  → P_base = T_camera_to_base @ [P_cam, 1]^T（ETH 矩阵）
  → rm_movej_p([x, y, z+hover_z, π, 0, 0], speed=25%)
    （夹爪竖直朝下姿态：rx=π, ry=0, rz=0）
  → 等待 0.5s 振动衰减
```

### Phase 3：Hand 相机 OpenCV 精定位 + EIH 夹取

```
采 7 帧 hand 相机图像
  → BGR → HSV → inRange（黑色盖阈值）
  → MORPH_OPEN + MORPH_CLOSE（去噪）
  → findContours → 圆形度过滤 → fitEllipse
  → 取 7 帧中位数像素中心 (u_med, v_med)
  
读当前末端 FK → T_base_end（4×4）
取深度 z = depth_at_point(u_med, v_med)
  → pixel_to_camera() → P_cam
  → P_fine = T_base_end @ T_cam_end @ [P_cam,1]^T（EIH）

rm_movej_p([xf,yf,zf+pre_z, π,0,0], speed=20%)  # 精调到 pre 位
开爪（openness=0.80）
rm_movel([xf,yf,zf+touch_z, π,0,0], speed=10%)   # 竖直下压
闭爪（openness=0.00）
rm_movel([xf,yf,zf+lift_z,  π,0,0], speed=20%)   # 竖直提起
```

---

## 关键参数调整

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grasp.hover_z` | 0.10 m | Phase2 悬停高度（应比试管高）|
| `grasp.pre_z` | 0.08 m | Phase3 精定位后悬停高度 |
| `grasp.touch_z` | 0.003 m | 下压接触量（太大会撞，太小夹不住）|
| `grasp.lift_z` | 0.15 m | 提起高度 |
| `hand_vision.hsv_lower/upper` | [0,0,0]/[180,80,70] | 盖颜色 HSV 范围，**需根据实际调整** |
| `hand_vision.n_frames` | 7 | 精定位融合帧数（越多越稳，越慢）|
| `yolo.conf` | 0.50 | YOLO 置信度阈值 |

### 调试 HSV 阈值的方法

```python
import cv2, numpy as np
img = cv2.imread("一帧hand相机图.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("盖中心 HSV:", hsv[y, x])   # y,x = 盖中心像素坐标
```

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError: best.pt` | 模型未放置 | 放到 `assets/model/best.pt` |
| `Phase1 未检测到试管盖` | conf 阈值过高 / 视野遮挡 | 降低 `yolo.conf` 或检查相机位置 |
| `Phase3 精定位失败` | HSV 阈值不匹配 | 调整 `hand_vision.hsv_upper/lower` |
| `Phase3 深度无效` | 相机距目标太近/远 | 检查 `depth.min_m/max_m` 范围 |
| `rm_movej_p 失败 code=1` | IK 无解（目标超出工作空间）| 检查 `hover_z` 是否合理 |
| `ModuleNotFoundError: Robotic_Arm` | 机械臂 SDK 未安装 | `pip install "Robotic_Arm>=1.0.0"` |

---

## 输出示例

```
══════════════════════════════════════════════════════════════
   tubeGrabber_v1  三阶段试管盖抓取 Demo（完全独立版）
══════════════════════════════════════════════════════════════

[Init] ✓ ETH: T_cam2base.json  shape=(4, 4)
[Init] ✓ EIH: T_cam2end.json  shape=(4, 4)
[Init] 连接机械臂 192.168.1.18:8080 ...
[Arm]  已连接 192.168.1.18:8080  handle=1
[Init] 开启 head 相机...
[head] 就绪  fx=612.3  serial=CPCS253000HM
[Init] 开启 hand 相机...
[hand] 就绪  fx=614.7  serial=CP84B4100090
[Init] ✓ 两路相机已就绪

[Phase1] 加载 YOLO: best.pt ...

[Phase1] ══════ 检测到 3 个试管盖 ══════
  编号        像素中心 (u, v)          深度       置信度
  ────────────────────────────────────────────────────
  [ 1]    (  312.4,   215.8)       z=0.482m    conf=0.921
  [ 2]    (  198.3,   301.2)       z=0.479m    conf=0.876
  [ 3]    (  421.7,   188.6)       z=0.485m    conf=0.754

请输入要抓取的试管编号 [1 ~ 3]: 2

[Phase2] ══════ ETH 3D 解算 ══════
  P_cam   = (-0.0821, +0.0342, +0.4790) m
  P_base  = (+0.2143, -0.1876, +0.0412) m

[Phase2] ══════ rm_movej_p → 上方 ══════
  目标: x=+0.2143 y=-0.1876 z=+0.1412  rx=π ry=0 rz=0
[Phase2] ✓ 到位，等待 0.5s 振动衰减...

[Phase3] ══════ Hand 相机精定位（7 帧融合）══════
  帧  1/7: 中心=(321.4, 248.7)
  帧  2/7: 中心=(320.8, 249.1)
  帧  3/7: 中心=(321.2, 248.9)
  ...
  中位数: (321.10, 248.90)  σ_x=0.24px  σ_y=0.18px  有效帧=7

[Phase3] ══════ EIH 3D 解算 ══════
  P_cam   = (+0.0041, +0.0018, +0.0823) m
  P_coarse= (+0.2143, -0.1876, +0.0412) m
  P_fine  = (+0.2138, -0.1882, +0.0408) m
  粗→精偏移: 7.2 mm

[Phase3] ══════ 精调 → pre 位 ══════
[Phase3] ══════ 开爪 (0.8) ══════
[Phase3] ══════ ↓ rm_movel 下压 ══════
[Phase3] ══════ 闭爪 (0.0) ══════
[Phase3] ══════ ↑ rm_movel 竖直提起 ══════
[Phase3] ✓ 夹取完成！

══════════════════════════════════════════════════════════════
   ✓  全流程完成！
══════════════════════════════════════════════════════════════
```
