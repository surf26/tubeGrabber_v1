"""
detect.py — Phase 1：Head 相机 YOLO 全局扫描

用俯视固定相机（head）拍一帧，YOLO 推理，
返回所有试管盖的像素位置和深度信息。
"""

import time
from pathlib import Path

from config import CFG, ROOT
from transforms import depth_at_bbox


def scan_tubes(head_cam) -> dict:
    """
    拍一帧 → YOLO 推理 → 按置信度降序编号。

    Returns:
        {1: {"bbox", "pixel", "confidence", "depth_m", "frame"}, ...}
    """
    yolo_cfg   = CFG["yolo"]
    model_path = (ROOT / yolo_cfg["model_path"]).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO 模型不存在: {model_path}\n"
            "  → 将 best.pt 放到 tubeGrabber_v1/assets/model/"
        )

    print(f"\n[Phase1] 加载 YOLO: {model_path.name} ...")
    from ultralytics import YOLO  # type: ignore
    yolo = YOLO(str(model_path))
    yolo.to(yolo_cfg["device"])

    # 等待相机第一帧
    frame = None
    for _ in range(20):
        frame = head_cam.grab()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        raise RuntimeError("[Phase1] head 相机无帧，请检查连接")

    results = yolo.predict(
        frame.color,
        conf    = yolo_cfg["conf"],
        iou     = yolo_cfg["iou"],
        imgsz   = yolo_cfg["imgsz"],
        verbose = False,
    )

    # 收集检测结果
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = bbox
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            z = depth_at_bbox(frame.depth, bbox)
            if z is None:
                print(f"  bbox 深度无效，跳过: {[round(b, 1) for b in bbox]}")
                continue
            detections.append({
                "bbox":       bbox,
                "pixel":      (u, v),
                "confidence": conf,
                "depth_m":    z,
                "frame":      frame,
            })

    # 按置信度降序排列
    detections.sort(key=lambda d: -d["confidence"])

    # 打印结果表格
    print(f"\n[Phase1] ══════ 检测到 {len(detections)} 个试管盖 ══════")
    print(f"  {'编号':>4}   {'像素中心 (u, v)':^22}   {'深度':>8}   {'置信度':>8}")
    print("  " + "─" * 52)

    tube_map = {}
    for i, d in enumerate(detections, 1):
        u, v = d["pixel"]
        print(f"  [{i:>2}]    ({u:>7.1f}, {v:>7.1f})       "
              f"z={d['depth_m']:.3f}m    conf={d['confidence']:.3f}")
        tube_map[i] = d

    return tube_map
