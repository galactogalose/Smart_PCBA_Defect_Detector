from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("     RESISTOR SEGMENTATION STARTED")
print("======================================\n")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = (
    PROJECT_ROOT
    / "data"
    / "originals"
    / "pcb_04.png"
)

csv_path = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotations.csv"
)

print("Project root :", PROJECT_ROOT)
print("Image path   :", image_path)
print("CSV path     :", csv_path)


# --------------------------------------------------
# 2. LOAD IMAGE
# --------------------------------------------------

print("\n[1/7] Loading PCB image...")

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not load image")
    exit()

print("✓ Image loaded successfully")
print("Image dimensions:", image.shape)


# --------------------------------------------------
# 3. FIND RESISTOR ANNOTATION
# --------------------------------------------------

print("\n[2/7] Reading resistor bounding box...")

resistor_box = None

with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if (
            row["image"] == image_path.name
            and row["component"] == "resistor"
        ):

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            resistor_box = (x, y, w, h)

            break


if resistor_box is None:
    print("ERROR: Resistor annotation not found")
    exit()


x, y, w, h = resistor_box

print("✓ Resistor annotation found")
print("Bounding box:", resistor_box)


# --------------------------------------------------
# 4. CROP RESISTOR ROI
# --------------------------------------------------

print("\n[3/7] Cropping resistor ROI...")

roi = image[
    y:y + h,
    x:x + w
].copy()

print("✓ Resistor ROI cropped")
print("ROI dimensions:", roi.shape)


# --------------------------------------------------
# 5. CONVERT TO GRAYSCALE
# --------------------------------------------------

print("\n[4/7] Converting ROI to grayscale...")

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

print("✓ Grayscale conversion completed")


# --------------------------------------------------
# 6. CREATE INITIAL MASK
# --------------------------------------------------

print("\n[5/7] Creating resistor mask...")

# Starting threshold only.
# We may need to tune this after seeing the result.

threshold_value = 120

_, mask = cv2.threshold(
    gray,
    threshold_value,
    255,
    cv2.THRESH_BINARY_INV
)

print("✓ Binary mask created")
print("Threshold:", threshold_value)


# --------------------------------------------------
# 7. CLEAN MASK
# --------------------------------------------------

print("\n[6/7] Cleaning mask...")

kernel = np.ones(
    (3, 3),
    np.uint8
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

print("✓ Mask cleaned")


# --------------------------------------------------
# 8. EXTRACT RESISTOR
# --------------------------------------------------

print("\n[7/7] Extracting resistor...")

extracted = cv2.bitwise_and(
    roi,
    roi,
    mask=mask
)

print("✓ Resistor extracted")

# --------------------------------------------------
# 9. DISPLAY RESULTS — ENLARGED FOR INSPECTION
# --------------------------------------------------

print("\n======================================")
print("     SEGMENTATION COMPLETED")
print("======================================")

print("\nOriginal ROI size :", roi.shape)
print("Enlarging preview only...")

# Scale factor ONLY for visualization
scale = 10

preview_roi = cv2.resize(
    roi,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST
)

preview_mask = cv2.resize(
    mask,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST
)

preview_extracted = cv2.resize(
    extracted,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST
)

print("✓ Preview enlarged by", scale, "times")
print("NOTE: Actual resistor data has NOT been resized.")

cv2.imshow(
    "1 - Resistor ROI - Enlarged Preview",
    preview_roi
)

cv2.imshow(
    "2 - Resistor Mask - Enlarged Preview",
    preview_mask
)

cv2.imshow(
    "3 - Extracted Resistor - Enlarged Preview",
    preview_extracted
)

print("\nLook carefully at:")
print("  WHITE = selected resistor pixels")
print("  BLACK = removed/background pixels")
print("\nPress ANY KEY to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✓ Preview closed")



print("\n======================================")
print("      PROGRAM FINISHED")
print("======================================\n")