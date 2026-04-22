"""
Data Augmentation Script for YOLO Training.

Reads images from data/characters/, applies augmentations (silhouette, color jitter,
rotation, flip), and saves results to data/yolo-training/images with corresponding labels.

Usage:
    python augment.py
    python augment.py --data-dir /path/to/data --augments-per-image 5
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_silhouette(img: Image.Image) -> Image.Image:
    """Convert image to black silhouette on white background."""
    gray = img.convert("L")
    threshold = 200
    silhouette = gray.point(lambda p: 0 if p < threshold else 255)
    return silhouette.convert("RGB")


def apply_color_jitter(img: Image.Image) -> Image.Image:
    """Randomly adjust brightness, contrast, saturation."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.4, 1.6))
    return img


def apply_rotation(img: Image.Image) -> Image.Image:
    """Rotate image by a random angle."""
    angle = random.uniform(-30, 30)
    return img.rotate(angle, expand=True, fillcolor=(128, 128, 128))


def apply_flip(img: Image.Image) -> Image.Image:
    """Randomly flip image horizontally."""
    if random.random() > 0.5:
        return ImageOps.mirror(img)
    return img


def apply_blur(img: Image.Image) -> Image.Image:
    """Apply slight Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))


AUGMENTATIONS = [
    ("silhouette", create_silhouette),
    ("color_jitter", apply_color_jitter),
    ("rotation", apply_rotation),
    ("flip", apply_flip),
    ("blur", apply_blur),
]


def create_yolo_label(class_id: int = 0) -> str:
    """Create a YOLO label assuming the object fills the entire image.

    class_id 0 = character, 1 = face.
    Returns: YOLO format line 'class_id cx cy w h' (normalized).
    """
    return f"{class_id} 0.5 0.5 1.0 1.0"


def augment_dataset(data_dir: str, augments_per_image: int):
    """Generate augmented images and labels for YOLO training."""
    characters_dir = os.path.join(data_dir, "characters")
    output_images_dir = os.path.join(data_dir, "yolo-training", "images")
    output_labels_dir = os.path.join(data_dir, "yolo-training", "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    if not os.path.exists(characters_dir):
        print(f"Error: Characters directory not found at {characters_dir}")
        return

    char_folders = [d for d in os.listdir(characters_dir)
                    if os.path.isdir(os.path.join(characters_dir, d))]
    if not char_folders:
        print(f"No character folders found in {characters_dir}")
        return

    total_generated = 0
    for char_name in char_folders:
        char_path = os.path.join(characters_dir, char_name)
        images = [f for f in os.listdir(char_path)
                  if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]

        if not images:
            print(f"No images found for character '{char_name}', skipping.")
            continue

        for img_file in images:
            img_path = os.path.join(char_path, img_file)
            img = Image.open(img_path).convert("RGB")
            stem = Path(img_file).stem

            # Save original
            orig_name = f"{char_name}_{stem}_orig"
            img.save(os.path.join(output_images_dir, f"{orig_name}.jpg"), "JPEG")
            label_line = create_yolo_label(class_id=0)
            with open(os.path.join(output_labels_dir, f"{orig_name}.txt"), "w") as f:
                f.write(label_line + "\n")
            total_generated += 1

            # Generate augmentations
            for i in range(augments_per_image):
                aug_img = img.copy()
                num_augs = random.randint(1, 3)
                selected = random.sample(AUGMENTATIONS, min(num_augs, len(AUGMENTATIONS)))
                for _, aug_fn in selected:
                    aug_img = aug_fn(aug_img)

                out_name = f"{char_name}_{stem}_aug{i}"
                aug_img.save(os.path.join(output_images_dir, f"{out_name}.jpg"), "JPEG")
                with open(os.path.join(output_labels_dir, f"{out_name}.txt"), "w") as f:
                    f.write(label_line + "\n")
                total_generated += 1

    print(f"Augmentation complete - generated {total_generated} images in {output_images_dir}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Augment character images for YOLO training")
    parser.add_argument("--data-dir", type=str, default=str(project_root / "data"),
                        help="Path to data directory")
    parser.add_argument("--augments-per-image", type=int, default=5,
                        help="Number of augmented images per original")
    args = parser.parse_args()

    augment_dataset(args.data_dir, args.augments_per_image)


if __name__ == "__main__":
    main()
