"""
YOLO Training Script for Character and Face Detection.

Trains YOLOv8 Nano on custom data and exports to ONNX format.
Supports local execution and Google Colab.

Usage:
    python train.py
    python train.py --data-dir /content/drive/MyDrive/ImageAnalyzer/data --models-dir /content/drive/MyDrive/ImageAnalyzer/models
"""

import argparse
import os
import yaml
from pathlib import Path
from ultralytics import YOLO


def generate_data_yaml(data_dir: str) -> str:
    """Generate data.yaml for YOLO training."""
    yaml_path = os.path.join(data_dir, "yolo-training", "data.yaml")
    data_config = {
        "path": os.path.join(data_dir, "yolo-training"),
        "train": "images",
        "val": "images",
        "nc": 2,
        "names": ["character", "face"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"Generated data.yaml at {yaml_path}")
    return yaml_path


def train(data_dir: str, models_dir: str, epochs: int, batch: int, imgsz: int):
    """Train YOLOv8n and export to ONNX."""
    yaml_path = generate_data_yaml(data_dir)

    images_dir = os.path.join(data_dir, "yolo-training", "images")
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print(f"Error: No training images found in {images_dir}")
        print("Run augment.py first to generate training data.")
        return

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=os.path.join(models_dir, "runs"),
        name="yolo-characters",
        exist_ok=True,
    )

    best_pt = os.path.join(models_dir, "runs", "yolo-characters", "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"Error: best.pt not found at {best_pt}")
        return

    trained_model = YOLO(best_pt)
    onnx_path = os.path.join(models_dir, "yolo-characters.onnx")
    trained_model.export(format="onnx", imgsz=imgsz)

    exported_onnx = best_pt.replace(".pt", ".onnx")
    if os.path.exists(exported_onnx):
        os.rename(exported_onnx, onnx_path)
        print(f"ONNX model saved to {onnx_path}")
    else:
        print(f"Warning: Expected ONNX at {exported_onnx}, check export output.")

    print("Training complete.")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Train YOLOv8 for character/face detection")
    parser.add_argument("--data-dir", type=str, default=str(project_root / "data"),
                        help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default=str(project_root / "models"),
                        help="Path to models directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    train(args.data_dir, args.models_dir, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
