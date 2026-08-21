from pathlib import Path
import cv2
import csv
import os

# -----------------------------
# 1. PROJECT PATHS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = PROJECT_ROOT / "data" / "originals" / "pcb_04.png"
annotation_folder = PROJECT_ROOT / "data" / "annotations"
csv_path = annotation_folder / "annotations.csv"

annotation_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. LOAD IMAGE
# -----------------------------
image = cv2.imread(str(image_path))

if image is None:
    print("Could not load image:", image_path)
    exit()

print("Image loaded successfully:", image_path.name)

# -----------------------------
# 3. COMPONENTS TO SELECT
# -----------------------------
components = ["IC", "resistor", "capacitor"]

annotations = []

# -----------------------------
# 4. SELECT ROI FOR EACH COMPONENT
# -----------------------------
for component in components:

    print(f"\nSelect the {component}")
    print("Drag a box around it, then press ENTER or SPACE.")

    window_name = f"Select {component}"

    x, y, w, h = cv2.selectROI(
        window_name,
        image,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(window_name)

    # If user cancels selection
    if w == 0 or h == 0:
        print(f"{component} selection cancelled.")
        continue

    annotations.append([
        image_path.name,
        component,
        int(x),
        int(y),
        int(w),
        int(h)
    ])

    print(
        f"{component} selected -> "
        f"x={x}, y={y}, width={w}, height={h}"
    )

cv2.destroyAllWindows()

# -----------------------------
# 5. SAVE TO CSV
# -----------------------------
file_exists = csv_path.exists()

with open(csv_path, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "image",
            "component",
            "x",
            "y",
            "width",
            "height"
        ])

    writer.writerows(annotations)

print("\nAnnotations saved to:")
print(csv_path)