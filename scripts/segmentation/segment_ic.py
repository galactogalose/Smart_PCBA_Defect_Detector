from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("       IC SEGMENTATION STARTED")
print("======================================\n")


# -----------------------------
# 1. PATHS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = PROJECT_ROOT / "data" / "originals" / "pcb_04.png"
csv_path = PROJECT_ROOT / "data" / "annotations" / "annotations.csv"

print("Project root :", PROJECT_ROOT)
print("Image path   :", image_path)
print("CSV path     :", csv_path)


# -----------------------------
# 2. LOAD ORIGINAL IMAGE
# -----------------------------
print("\n[1/8] Loading original PCB image...")

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not load image")
    exit()

print("✓ Image loaded")
print("Original dimensions:", image.shape)


# -----------------------------
# 3. READ IC ANNOTATION
# -----------------------------
print("\n[2/8] Reading IC bounding box...")

ic_box = None

with open(csv_path, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if (
            row["image"] == image_path.name
            and row["component"] == "IC"
        ):
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            ic_box = (x, y, w, h)
            break


if ic_box is None:
    print("ERROR: IC annotation not found")
    exit()


x, y, w, h = ic_box

print("✓ IC annotation found")
print("Bounding box:", ic_box)


# -----------------------------
# 4. CROP ROI FROM ORIGINAL IMAGE
# -----------------------------
print("\n[3/8] Cropping IC ROI...")

roi = image[
    y:y+h,
    x:x+w
].copy()

print("✓ ROI cropped")
print("ROI dimensions:", roi.shape)


# -----------------------------
# 5. GRAYSCALE
# -----------------------------
print("\n[4/8] Converting ROI to grayscale...")

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

print("✓ Grayscale conversion done")


# -----------------------------
# 6. OTSU THRESHOLD
# -----------------------------
print("\n[5/8] Creating adaptive mask with Otsu...")

otsu_value, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print("✓ Otsu thresholding done")
print("Selected threshold:", otsu_value)


# -----------------------------
# 7. MORPHOLOGICAL CLEANUP
# -----------------------------
print("\n[6/8] Cleaning mask...")

kernel = np.ones((3, 3), np.uint8)

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

print("✓ Mask cleaned")


# -----------------------------
# 8. KEEP LARGEST CONTOUR
# -----------------------------
print("\n[7/8] Selecting largest IC-like contour...")

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("ERROR: No contours found")
    exit()


largest_contour = max(
    contours,
    key=cv2.contourArea
)

clean_mask = np.zeros_like(mask)

cv2.drawContours(
    clean_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED
)

mask = clean_mask

print("✓ Largest contour selected")
print("Contour area:", cv2.contourArea(largest_contour))


# -----------------------------
# 9. EXTRACT IC
# -----------------------------
print("\n[8/8] Extracting IC...")

extracted = cv2.bitwise_and(
    roi,
    roi,
    mask=mask
)

print("✓ IC extracted")


# -----------------------------
# 10. DISPLAY-SIZED PREVIEW ONLY
# -----------------------------

# This scaling is ONLY for viewing.
# It does not affect the mask or coordinates.

max_display_width = 900
max_display_height = 700

roi_h, roi_w = roi.shape[:2]

scale_w = max_display_width / roi_w
scale_h = max_display_height / roi_h

display_scale = min(
    scale_w,
    scale_h,
    3.0
)

preview_w = int(roi_w * display_scale)
preview_h = int(roi_h * display_scale)

preview_roi = cv2.resize(
    roi,
    (preview_w, preview_h),
    interpolation=cv2.INTER_NEAREST
)

preview_mask = cv2.resize(
    mask,
    (preview_w, preview_h),
    interpolation=cv2.INTER_NEAREST
)

preview_extracted = cv2.resize(
    extracted,
    (preview_w, preview_h),
    interpolation=cv2.INTER_NEAREST
)


# -----------------------------
# 11. DISPLAY
# -----------------------------
print("\nOpening segmentation previews...")

cv2.imshow(
    "1 - IC ROI",
    preview_roi
)

cv2.imshow(
    "2 - IC Mask",
    preview_mask
)

cv2.imshow(
    "3 - Extracted IC",
    preview_extracted
)

print("\nWHITE = selected component")
print("BLACK = background")
print("Press ANY KEY to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()


print("\n======================================")
print("       IC SEGMENTATION FINISHED")
print("======================================\n")