from pathlib import Path
import cv2
import csv


# -----------------------------
# 1. PROJECT PATHS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = PROJECT_ROOT / "data" / "originals" / "pcb_04.png"
annotation_folder = PROJECT_ROOT / "data" / "annotations"
csv_path = annotation_folder / "annotations.csv"

annotation_folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. LOAD ORIGINAL IMAGE
# -----------------------------
image = cv2.imread(str(image_path))

if image is None:
    print("Could not load image:", image_path)
    exit()

original_h, original_w = image.shape[:2]

print("Image loaded successfully:", image_path.name)
print("Original size:", original_w, "x", original_h)


# -----------------------------
# 3. CREATE DISPLAY-SIZED COPY
# -----------------------------

# Maximum size for annotation window
max_display_width = 1200
max_display_height = 750

# Calculate scale required to fit image on screen
scale_w = max_display_width / original_w
scale_h = max_display_height / original_h

# Never enlarge small images
scale = min(scale_w, scale_h, 1.0)

display_w = int(original_w * scale)
display_h = int(original_h * scale)

display_image = cv2.resize(
    image,
    (display_w, display_h),
    interpolation=cv2.INTER_AREA
)

print("Display size :", display_w, "x", display_h)
print("Display scale:", round(scale, 4))


# -----------------------------
# 4. COMPONENTS TO SELECT
# -----------------------------
components = [
    "IC",
    "resistor",
    "capacitor"
]

annotations = []


# -----------------------------
# 5. SELECT ROI FOR EACH COMPONENT
# -----------------------------
for component in components:

    print(f"\nSelect the {component}")
    print("Drag a box around it, then press ENTER or SPACE.")

    window_name = f"Select {component}"

    # ROI is selected on the RESIZED DISPLAY IMAGE
    x_display, y_display, w_display, h_display = cv2.selectROI(
        window_name,
        display_image,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(window_name)

    # If user cancels selection
    if w_display == 0 or h_display == 0:
        print(f"{component} selection cancelled.")
        continue


    # -----------------------------
    # 6. CONVERT COORDINATES BACK
    #    TO ORIGINAL IMAGE SIZE
    # -----------------------------

    x = int(x_display / scale)
    y = int(y_display / scale)

    w = int(w_display / scale)
    h = int(h_display / scale)


    annotations.append([
        image_path.name,
        component,
        x,
        y,
        w,
        h
    ])

    print(f"{component} selected")

    print(
        "Display bbox ->",
        x_display,
        y_display,
        w_display,
        h_display
    )

    print(
        "Original bbox ->",
        x,
        y,
        w,
        h
    )


cv2.destroyAllWindows()


# -----------------------------
# 7. SAVE TO CSV
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