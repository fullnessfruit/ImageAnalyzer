"""
Export pretrained YOLOv8n (COCO) to ONNX for person detection.

Run this if the auto-download of yolo-person.onnx fails.

Usage:
    cd training
    python export_person_model.py
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")
result = model.export(format="onnx", imgsz=640)

exported = Path(result)
dest = models_dir / "yolo-person.onnx"
if exported.exists():
    shutil.move(str(exported), str(dest))
    print(f"Person detection model saved to {dest}")
else:
    print(f"Export failed: expected ONNX at {exported}")
