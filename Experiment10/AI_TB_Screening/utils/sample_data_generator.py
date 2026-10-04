"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Sample / Starter Dataset Generator
=============================================================================
Generates anatomically inspired synthetic chest X-ray images (Normal vs TB)
to enable immediate out-of-the-box training, testing, and college demonstrations
without requiring immediate multi-gigabyte downloads.

If external real-world X-rays (such as the Kaggle TB Chest X-ray Database)
are placed in dataset/TB and dataset/Normal, the entire pipeline processes
them seamlessly.
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def draw_chest_anatomy(width=512, height=512, is_tb=False, seed=None):
    """
    Renders an anatomically inspired chest radiograph matrix.
    Includes thoracic cage, mediastinum, bilateral lung fields, clavicles,
    ribs, diaphragmatic domes, and pathology features if is_tb=True.
    """
    rng = np.random.default_rng(seed)

    # 1. Base dark background with slight detector noise
    img = np.zeros((height, width), dtype=np.float32)
    img += rng.normal(15, 3, (height, width))

    # 2. Thorax silhouette (soft tissue contour)
    cv2.ellipse(img, (width // 2, int(height * 0.62)), (int(width * 0.42), int(height * 0.48)),
                0, 0, 360, 55, -1)

    # 3. Bilateral lung fields (radiolucent / darker areas)
    # Right lung (viewer's left)
    cv2.ellipse(img, (int(width * 0.33), int(height * 0.52)), (int(width * 0.15), int(height * 0.28)),
                -4, 0, 360, 22, -1)
    # Left lung (viewer's right)
    cv2.ellipse(img, (int(width * 0.67), int(height * 0.52)), (int(width * 0.14), int(height * 0.28)),
                4, 0, 360, 22, -1)

    # 4. Spine (central radiopaque vertebral column)
    spine_x = width // 2
    cv2.rectangle(img, (spine_x - 14, int(height * 0.1)), (spine_x + 14, int(height * 0.9)), 85, -1)
    for y in range(int(height * 0.15), int(height * 0.85), 24):
        cv2.line(img, (spine_x - 18, y), (spine_x + 18, y), 110, 2)

    # 5. Diaphragm domes & costophrenic angles
    cv2.ellipse(img, (int(width * 0.33), int(height * 0.82)), (int(width * 0.16), int(height * 0.12)),
                0, 0, 180, 115, -1)
    cv2.ellipse(img, (int(width * 0.67), int(height * 0.84)), (int(width * 0.16), int(height * 0.11)),
                0, 0, 180, 110, -1)

    # 6. Cardiac silhouette (heart shadow, prominent on anatomical left / viewer's right)
    cv2.ellipse(img, (int(width * 0.55), int(height * 0.66)), (int(width * 0.15), int(height * 0.16)),
                25, 0, 360, 120, -1)

    # 7. Clavicles (collar bones)
    # Right clavicle
    pts_r = np.array([[int(width * 0.45), int(height * 0.26)],
                      [int(width * 0.32), int(height * 0.23)],
                      [int(width * 0.18), int(height * 0.25)]], np.int32)
    cv2.polylines(img, [pts_r], False, 130, 7)
    # Left clavicle
    pts_l = np.array([[int(width * 0.55), int(height * 0.26)],
                      [int(width * 0.68), int(height * 0.23)],
                      [int(width * 0.82), int(height * 0.25)]], np.int32)
    cv2.polylines(img, [pts_l], False, 130, 7)

    # 8. Rib arches (posterior and anterior ribs)
    for i, offset_y in enumerate(range(int(height * 0.3), int(height * 0.75), 32)):
        # Right ribs
        cv2.ellipse(img, (int(width * 0.28), offset_y), (int(width * 0.18), 16),
                    12, 0, 180, 70 + (i * 3), 4)
        # Left ribs
        cv2.ellipse(img, (int(width * 0.72), offset_y), (int(width * 0.18), 16),
                    -12, 0, 180, 70 + (i * 3), 4)

    # 9. Bronchovascular markings (delicate vascular arborization)
    for _ in range(25):
        # Right lung vessels
        rx1 = int(width * 0.42 + rng.integers(-10, 10))
        ry1 = int(height * 0.50 + rng.integers(-20, 20))
        rx2 = int(rx1 - rng.integers(30, 80))
        ry2 = int(ry1 + rng.integers(-40, 40))
        cv2.line(img, (rx1, ry1), (rx2, ry2), 45, 1)

        # Left lung vessels
        lx1 = int(width * 0.58 + rng.integers(-10, 10))
        ly1 = int(height * 0.50 + rng.integers(-20, 20))
        lx2 = int(lx1 + rng.integers(30, 80))
        ly2 = int(ly1 + rng.integers(-40, 40))
        cv2.line(img, (lx1, ly1), (lx2, ly2), 45, 1)

    # 10. If TB: simulate classic radiographic manifestations
    # (Apical infiltrates, fibro-cavitary opacities, or consolidation)
    if is_tb:
        lesion_type = rng.choice(["apical_infiltrate", "cavitary", "consolidation"])

        # Primarily in upper lobes (apices), occasionally bilateral
        side_choice = rng.choice(["right", "left", "both"], p=[0.45, 0.35, 0.20])
        sides = ["right"] if side_choice == "right" else (["left"] if side_choice == "left" else ["right", "left"])

        for side in sides:
            cx = int(width * 0.31) if side == "right" else int(width * 0.69)
            cy = int(height * 0.35 + rng.integers(-25, 25))

            if lesion_type == "apical_infiltrate":
                # Patchy mottled cloud of opacity
                for _ in range(12):
                    px = int(cx + rng.normal(0, 15))
                    py = int(cy + rng.normal(0, 15))
                    rad = int(rng.integers(8, 22))
                    intensity = int(rng.integers(80, 140))
                    cv2.circle(img, (px, py), rad, intensity, -1)

            elif lesion_type == "cavitary":
                # Cavitary lesion: radiopaque thick rim with darker radiolucent interior
                rad = int(rng.integers(22, 35))
                cv2.circle(img, (cx, cy), rad, 125, -1)
                cv2.circle(img, (cx, cy), int(rad * 0.6), 35, -1)

            elif lesion_type == "consolidation":
                # Dense lobar/segmental consolidation
                pts = []
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    r = rng.integers(25, 45)
                    pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
                cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 110)

    # 11. Soft radiographic blurring and anatomical tissue blending
    img = cv2.GaussianBlur(img, (9, 9), 3.0)

    # 12. Fine radiographic quantum mottle / detector noise
    noise = rng.normal(0, 4.0, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Convert to 3-channel RGB image
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def generate_starter_dataset(num_per_class: int = 30, overwrite: bool = False) -> dict:
    """
    Populates dataset/Normal and dataset/TB with verified starter chest X-ray images.
    Returns status dictionary with created image counts.
    """
    config.TB_DIR.mkdir(parents=True, exist_ok=True)
    config.NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    existing_tb = list(config.TB_DIR.glob("*.jpg")) + list(config.TB_DIR.glob("*.png"))
    existing_normal = list(config.NORMAL_DIR.glob("*.jpg")) + list(config.NORMAL_DIR.glob("*.png"))

    if not overwrite and (len(existing_tb) >= num_per_class and len(existing_normal) >= num_per_class):
        return {
            "status": "skipped",
            "message": f"Dataset already contains {len(existing_tb)} TB and {len(existing_normal)} Normal images.",
            "tb_count": len(existing_tb),
            "normal_count": len(existing_normal)
        }

    # Generate Normal images
    print(f"Generating {num_per_class} sample Normal chest X-rays...")
    for i in range(1, num_per_class + 1):
        filename = f"Normal_{i:04d}.jpg"
        img = draw_chest_anatomy(width=512, height=512, is_tb=False, seed=1000 + i)
        filepath = config.NORMAL_DIR / filename
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Generate TB images
    print(f"Generating {num_per_class} sample TB chest X-rays...")
    for i in range(1, num_per_class + 1):
        filename = f"TB_{i:04d}.jpg"
        img = draw_chest_anatomy(width=512, height=512, is_tb=True, seed=2000 + i)
        filepath = config.TB_DIR / filename
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Starter dataset ready: {num_per_class} Normal, {num_per_class} TB images.")
    return {
        "status": "success",
        "message": f"Generated {num_per_class} Normal and {num_per_class} TB starter chest X-ray images.",
        "tb_count": num_per_class,
        "normal_count": num_per_class
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate starter chest X-ray dataset for TB screening.")
    parser.add_argument("--count", type=int, default=30, help="Number of images per class (default: 30)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing images")
    args = parser.parse_args()

    result = generate_starter_dataset(num_per_class=args.count, overwrite=args.force)
    print(result)
