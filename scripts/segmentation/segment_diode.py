from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("      DIODE SEGMENTATION STARTED")
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
    print("✗ ERROR: Could not load image")
    exit()

print("✓ Image loaded successfully")
print("  Dimensions:", image.shape)


# --------------------------------------------------
# 3. FIND DIODE ANNOTATION
# --------------------------------------------------

print("\n[2/7] Reading diode bounding box...")

diode_box = None

with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if (
            row["image"] == image_path.name
            and row["component"] == "diode"
        ):

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            diode_box = (x, y, w, h)
            break


if diode_box is None:
    print("✗ ERROR: Diode annotation not found")
    exit()


x, y, w, h = diode_box

print("✓ Diode annotation found")
print("  Bounding box:", diode_box)


# --------------------------------------------------
# 4. CROP DIODE ROI
# --------------------------------------------------

print("\n[3/7] Cropping diode ROI...")

roi = image[
    y:y + h,
    x:x + w
].copy()

print("✓ Diode ROI cropped")
print("  ROI dimensions:", roi.shape)


# --------------------------------------------------
# 5. CONVERT TO GRAYSCALE
# --------------------------------------------------

print("\n[4/7] Converting diode ROI to grayscale...")

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

print("✓ Grayscale conversion completed")


# --------------------------------------------------
# 6. OTSU THRESHOLDING
# --------------------------------------------------

print("\n[5/7] Creating adaptive mask using Otsu...")

otsu_value, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print("✓ Otsu mask created")
print("  Threshold:", otsu_value)


# --------------------------------------------------
# 7. CLEAN MASK
# --------------------------------------------------

print("\n[6/7] Cleaning diode mask...")

kernel = np.ones(
    (3, 3),
    np.uint8
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

print("✓ Initial mask cleaned")


# --------------------------------------------------
# 8. KEEP LARGEST CONTOUR
# --------------------------------------------------

print("\nFinding largest diode-like contour...")

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("✗ ERROR: No diode contour found")
    exit()


largest_contour = max(
    contours,
    key=cv2.contourArea
)

largest_area = cv2.contourArea(
    largest_contour
)

print("✓ Largest contour selected")
print("  Contour area:", largest_area)


clean_mask = np.zeros_like(mask)

cv2.drawContours(
    clean_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED
)


clean_mask = cv2.dilate(
    clean_mask,
    np.ones((3, 3), np.uint8),
    iterations=1
)

mask = clean_mask

print("✓ Final diode mask created")


# --------------------------------------------------
# 9. EXTRACT DIODE
# --------------------------------------------------

print("\n[7/7] Extracting diode...")

extracted = cv2.bitwise_and(
    roi,
    roi,
    mask=mask
)

print("✓ Diode extracted successfully")


# --------------------------------------------------
# 10. ENLARGE PREVIEW
# --------------------------------------------------

scale = 6

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


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n======================================")
print("      SEGMENTATION COMPLETED")
print("======================================")

print("\nOpening:")
print("1 - Diode ROI")
print("2 - Diode Mask")
print("3 - Extracted Diode")

print("\nWHITE = diode")
print("BLACK = background")
print("\nPress ANY KEY to close.")


cv2.imshow(
    "1 - Diode ROI",
    preview_roi
)

cv2.imshow(
    "2 - Diode Mask",
    preview_mask
)

cv2.imshow(
    "3 - Extracted Diode",
    preview_extracted
)

cv2.waitKey(0)

cv2.destroyAllWindows()


print("\n======================================")
print("       PROGRAM FINISHED")
print("======================================\n")