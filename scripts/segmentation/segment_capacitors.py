from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("   CAPACITOR SEGMENTATION STARTED")
print("======================================\n")


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

PCB_NAME = "pcb_04"
COMPONENT_NAME = "capacitor"


# --------------------------------------------------
# 2. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = (
    PROJECT_ROOT
    / "data"
    / "originals"
    / f"{PCB_NAME}.png"
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
# 3. LOAD ORIGINAL IMAGE
# --------------------------------------------------

print("\n[1/8] Loading original PCB image...")

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not load image")
    print("Checked:", image_path)
    exit()

print("✓ Image loaded successfully")
print("Image dimensions:", image.shape)


# --------------------------------------------------
# 4. READ CAPACITOR ANNOTATION
# --------------------------------------------------

print("\n[2/8] Reading capacitor bounding box...")

capacitor_box = None

with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if (
            row["image"] == image_path.name
            and row["component"] == COMPONENT_NAME
        ):

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            capacitor_box = (x, y, w, h)

            break


if capacitor_box is None:
    print("ERROR: Capacitor annotation not found")
    print("Image searched:", image_path.name)
    exit()


x, y, w, h = capacitor_box

print("✓ Capacitor annotation found")
print("Bounding box:", capacitor_box)

print("x      :", x)
print("y      :", y)
print("width  :", w)
print("height :", h)


# --------------------------------------------------
# 5. CROP CAPACITOR ROI
# --------------------------------------------------

print("\n[3/8] Cropping capacitor ROI...")

roi = image[
    y:y + h,
    x:x + w
].copy()

print("✓ Capacitor ROI cropped")
print("ROI dimensions:", roi.shape)


# --------------------------------------------------
# 6. CONVERT ROI TO GRAYSCALE
# --------------------------------------------------

print("\n[4/8] Converting capacitor ROI to grayscale...")

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

print("✓ Grayscale conversion completed")


# --------------------------------------------------
# 7. OTSU THRESHOLDING
# --------------------------------------------------

print("\n[5/8] Creating adaptive binary mask with Otsu...")

# Otsu automatically determines the threshold
# for this particular capacitor ROI.

otsu_value, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print("✓ Otsu mask created")
print("Automatically selected threshold:", otsu_value)


# --------------------------------------------------
# 8. CLEAN MASK
# --------------------------------------------------

print("\n[6/8] Cleaning capacitor mask...")

kernel = np.ones(
    (3, 3),
    np.uint8
)

# Remove tiny white noise
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# Fill small holes
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

print("✓ Initial mask cleaned")


# --------------------------------------------------
# 9. KEEP LARGEST CONTOUR
# --------------------------------------------------

print("\n[7/8] Selecting largest capacitor-like contour...")

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

largest_area = cv2.contourArea(
    largest_contour
)

print("✓ Contours found:", len(contours))
print("Largest contour area:", largest_area)


# Create clean mask containing only
# the largest detected object

clean_mask = np.zeros_like(mask)

cv2.drawContours(
    clean_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED
)


# Slightly expand the selected component
# so edge pixels are not lost

clean_mask = cv2.dilate(
    clean_mask,
    np.ones((3, 3), np.uint8),
    iterations=1
)

mask = clean_mask

print("✓ Final capacitor mask created")


# --------------------------------------------------
# 10. EXTRACT CAPACITOR
# --------------------------------------------------

print("\n[8/8] Extracting capacitor...")

extracted = cv2.bitwise_and(
    roi,
    roi,
    mask=mask
)

print("✓ Capacitor extracted successfully")


# --------------------------------------------------
# 11. CREATE CONTOUR PREVIEW
# --------------------------------------------------

contour_preview = roi.copy()

cv2.drawContours(
    contour_preview,
    [largest_contour],
    -1,
    (0, 255, 0),
    2
)


# --------------------------------------------------
# 12. RESIZE PREVIEW ONLY
# --------------------------------------------------

# This is only for viewing.
# Actual ROI/mask dimensions remain unchanged.

max_display_width = 900
max_display_height = 700

roi_h, roi_w = roi.shape[:2]

scale_w = max_display_width / roi_w
scale_h = max_display_height / roi_h

display_scale = min(
    scale_w,
    scale_h,
    4.0
)

preview_w = int(
    roi_w * display_scale
)

preview_h = int(
    roi_h * display_scale
)


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

preview_contour = cv2.resize(
    contour_preview,
    (preview_w, preview_h),
    interpolation=cv2.INTER_NEAREST
)


# --------------------------------------------------
# 13. DISPLAY RESULTS
# --------------------------------------------------

print("\n======================================")
print("   CAPACITOR SEGMENTATION COMPLETED")
print("======================================")

print("\nOpening four windows:")
print("1 - Capacitor ROI")
print("2 - Capacitor Mask")
print("3 - Extracted Capacitor")
print("4 - Selected Contour")

print("\nExpected:")
print("WHITE = capacitor")
print("BLACK = PCB/background")

print("\nClick any OpenCV window")
print("and press ANY KEY to close.")


cv2.imshow(
    "1 - Capacitor ROI",
    preview_roi
)

cv2.imshow(
    "2 - Capacitor Mask",
    preview_mask
)

cv2.imshow(
    "3 - Extracted Capacitor",
    preview_extracted
)

cv2.imshow(
    "4 - Capacitor Contour",
    preview_contour
)


cv2.waitKey(0)

print("✓ Key pressed")
print("Closing windows...")

cv2.destroyAllWindows()


# --------------------------------------------------
# 14. FINISHED
# --------------------------------------------------

print("\n======================================")
print("       PROGRAM FINISHED")
print("======================================\n")