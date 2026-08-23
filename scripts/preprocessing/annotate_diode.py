from pathlib import Path
import cv2
import csv


print("\n======================================")
print("       DIODE ANNOTATION STARTED")
print("======================================\n")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = (
    PROJECT_ROOT
    / "data"
    / "originals"
    / "pcb_09.png"
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

print("\n[1/4] Loading PCB image...")

image = cv2.imread(str(image_path))

if image is None:
    print("✗ ERROR: Could not load image")
    exit()

print("✓ Image loaded successfully")
print("  Dimensions:", image.shape)


# --------------------------------------------------
# 3. SELECT DIODE ROI
# --------------------------------------------------

print("\n[2/4] Select ONE diode")
print("Drag a tight box around ONE orange diode.")
print("Then press ENTER or SPACE.")

window_name = "Select Diode"

x, y, w, h = cv2.selectROI(
    window_name,
    image,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyWindow(window_name)


if w == 0 or h == 0:
    print("✗ Diode selection cancelled")
    cv2.destroyAllWindows()
    exit()


print("✓ Diode selected")
print("  x      :", x)
print("  y      :", y)
print("  width  :", w)
print("  height :", h)


# --------------------------------------------------
# 4. SAVE ANNOTATION TO CSV
# --------------------------------------------------

print("\n[3/4] Saving diode annotation...")

file_exists = csv_path.exists()

with open(
    csv_path,
    "a",
    newline=""
) as file:

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

    writer.writerow([
        image_path.name,
        "diode",
        int(x),
        int(y),
        int(w),
        int(h)
    ])


print("✓ Diode annotation saved")


# --------------------------------------------------
# 5. SHOW SELECTED BOX FOR VERIFICATION
# --------------------------------------------------

print("\n[4/4] Showing annotation preview...")

preview = image.copy()

cv2.rectangle(
    preview,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

cv2.putText(
    preview,
    "diode",
    (x, max(y - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imshow(
    "Diode Annotation Preview",
    preview
)

print("\nClick on the image window.")
print("Press ANY KEY to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()


print("\n======================================")
print("       DIODE ANNOTATION DONE")
print("======================================\n")