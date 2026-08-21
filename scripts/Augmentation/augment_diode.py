from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("      DIODE AUGMENTATION STARTED")
print("======================================\n")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

print("[STEP 1/10] Setting up paths...")

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

output_folder = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / "pcb_01"
    / "diode"
)

print("Output folder:", output_folder)
print("Folder exists:", output_folder.exists())

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

metadata_path = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / "augmentation_metadata.csv"
)

print("✓ Paths configured")
print("  Project root :", PROJECT_ROOT)
print("  Image        :", image_path)
print("  CSV          :", csv_path)
print("  Output       :", output_folder)


# --------------------------------------------------
# 2. LOAD ORIGINAL PCB
# --------------------------------------------------

print("\n[STEP 2/10] Loading original PCB...")

image = cv2.imread(str(image_path))

if image is None:
    print("✗ ERROR: Could not load PCB")
    print("  Checked:", image_path)
    exit()

print("✓ PCB loaded successfully")
print("  Dimensions:", image.shape)


# --------------------------------------------------
# 3. READ DIODE ANNOTATION
# --------------------------------------------------

print("\n[STEP 3/10] Searching for diode annotation...")

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
    print("  Image searched:", image_path.name)
    exit()


print("✓ Diode annotation found")
print("  Bounding box:", diode_box)

x, y, w, h = diode_box

print("  x      :", x)
print("  y      :", y)
print("  width  :", w)
print("  height :", h)


# --------------------------------------------------
# 4. CROP ROI
# --------------------------------------------------

print("\n[STEP 4/10] Cropping diode ROI...")

roi = image[
    y:y + h,
    x:x + w
].copy()

print("✓ ROI cropped")
print("  ROI dimensions:", roi.shape)


# --------------------------------------------------
# 5. SEGMENT BLACK DIODE
# --------------------------------------------------

print("\n[STEP 5/10] Segmenting black diode...")

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

print("✓ ROI converted to grayscale")


otsu_value, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print("✓ Otsu mask created")
print("  Threshold:", otsu_value)


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

print("✓ Largest diode-like contour selected")
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


component = cv2.bitwise_and(
    roi,
    roi,
    mask=mask
)

print("✓ Black diode segmented successfully")

# --------------------------------------------------
# 6. REMOVE ORIGINAL DIODE
# --------------------------------------------------

print("\n[STEP 6/10] Removing original diode from PCB...")

full_mask = np.zeros(
    image.shape[:2],
    dtype=np.uint8
)

full_mask[
    y:y + h,
    x:x + w
] = mask


# Dilate a little so the whole diode region is removed
inpaint_kernel = np.ones(
    (5, 5),
    np.uint8
)

full_mask = cv2.dilate(
    full_mask,
    inpaint_kernel,
    iterations=1
)


background = cv2.inpaint(
    image,
    full_mask,
    3,
    cv2.INPAINT_TELEA
)

print("✓ Original diode region removed")
print("✓ Background reconstructed")


# --------------------------------------------------
# 7A. HELPER FUNCTION: PASTE COMPONENT
# --------------------------------------------------

def paste_component(
    base_image,
    component_img,
    component_mask,
    new_x,
    new_y
):

    result = base_image.copy()

    comp_h, comp_w = component_img.shape[:2]

    # Make sure it stays inside image boundaries
    if (
        new_x < 0
        or new_y < 0
        or new_x + comp_w > result.shape[1]
        or new_y + comp_h > result.shape[0]
    ):
        print("✗ ERROR: Component would go outside PCB boundary")
        return None


    destination = result[
        new_y:new_y + comp_h,
        new_x:new_x + comp_w
    ]


    inverse_mask = cv2.bitwise_not(
        component_mask
    )


    destination_background = cv2.bitwise_and(
        destination,
        destination,
        mask=inverse_mask
    )


    foreground = cv2.bitwise_and(
        component_img,
        component_img,
        mask=component_mask
    )


    combined = cv2.add(
        destination_background,
        foreground
    )


    result[
        new_y:new_y + comp_h,
        new_x:new_x + comp_w
    ] = combined

    return result


# --------------------------------------------------
# 7B. HELPER FUNCTION: ROTATE COMPONENT
# --------------------------------------------------

def rotate_component(
    component_img,
    component_mask,
    angle
):

    comp_h, comp_w = component_img.shape[:2]

    center = (
        comp_w // 2,
        comp_h // 2
    )


    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )


    rotated_img = cv2.warpAffine(
        component_img,
        rotation_matrix,
        (comp_w, comp_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0)
    )


    rotated_mask = cv2.warpAffine(
        component_mask,
        rotation_matrix,
        (comp_w, comp_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )


    return rotated_img, rotated_mask


# --------------------------------------------------
# 7C. HELPER FUNCTION: SAVE RESULT
# --------------------------------------------------

metadata_rows = []


def save_result(
    result,
    filename,
    augmentation_type,
    parameter,
    new_x,
    new_y
):

    if result is None:
        print("✗ Result not saved because augmentation failed")
        return
    
   
    output_path = output_folder / filename

    saved = cv2.imwrite(
        str(output_path),
        result
    )


    if saved:
        print("✓ Saved:", filename)
    else:
        print("✗ ERROR: Could not save:", filename)


    metadata_rows.append([
        image_path.name,
        filename,
        "diode",
        augmentation_type,
        parameter,
        x,
        y,
        new_x,
        new_y,
        w,
        h
    ])


# --------------------------------------------------
# 7. GENERATE AUGMENTATIONS
# --------------------------------------------------

print("\n======================================")
print("       GENERATING AUGMENTATIONS")
print("======================================")


# --------------------------------------------------
# AUGMENTATION 1/5 — ROTATE +5°
# --------------------------------------------------

print("\n[AUGMENTATION 1/5]")
print("→ Rotating diode by +5 degrees...")

rot5_img, rot5_mask = rotate_component(
    component,
    mask,
    5
)

print("✓ Rotation completed")

result_rot5 = paste_component(
    background,
    rot5_img,
    rot5_mask,
    x,
    y
)

print("✓ Component pasted onto PCB")

save_result(
    result_rot5,
    "pcb_04_diode_rotate_05.png",
    "rotate",
    "+5 degrees",
    x,
    y
)


# --------------------------------------------------
# AUGMENTATION 2/5 — HORIZONTAL FLIP
# --------------------------------------------------

print("\n[AUGMENTATION 2/5]")
print("→ Flipping diode horizontally...")

flip_img = cv2.flip(
    component,
    1
)

flip_mask = cv2.flip(
    mask,
    1
)

print("✓ Flip completed")

result_flip = paste_component(
    background,
    flip_img,
    flip_mask,
    x,
    y
)

print("✓ Component pasted onto PCB")

save_result(
    result_flip,
    "pcb_04_diode_flip_horizontal.png",
    "flip",
    "horizontal",
    x,
    y
)


# --------------------------------------------------
# AUGMENTATION 3/5 — MOVE +10 PX
# --------------------------------------------------

print("\n[AUGMENTATION 3/5]")
print("→ Moving diode by +10 pixels...")

new_x_10 = x + 10
new_y_10 = y

print("  Original position:", (x, y))
print("  New position     :", (new_x_10, new_y_10))

result_move10 = paste_component(
    background,
    component,
    mask,
    new_x_10,
    new_y_10
)

print("✓ Movement completed")
print("✓ Component pasted onto PCB")

save_result(
    result_move10,
    "pcb_04_diode_move_10px.png",
    "move",
    "+10 px x-direction",
    new_x_10,
    new_y_10
)


# --------------------------------------------------
# AUGMENTATION 4/5 — MOVE +25 PX
# --------------------------------------------------

print("\n[AUGMENTATION 4/5]")
print("→ Moving diode by +25 pixels...")

new_x_25 = x + 25
new_y_25 = y

print("  Original position:", (x, y))
print("  New position     :", (new_x_25, new_y_25))

result_move25 = paste_component(
    background,
    component,
    mask,
    new_x_25,
    new_y_25
)

print("✓ Movement completed")
print("✓ Component pasted onto PCB")

save_result(
    result_move25,
    "pcb_04_diode_move_25px.png",
    "move",
    "+25 px x-direction",
    new_x_25,
    new_y_25
)


# --------------------------------------------------
# AUGMENTATION 5/5 — ROTATE +15°
# --------------------------------------------------

print("\n[AUGMENTATION 5/5]")
print("→ Rotating diode by +15 degrees...")

rot15_img, rot15_mask = rotate_component(
    component,
    mask,
    15
)

print("✓ Rotation completed")

result_rot15 = paste_component(
    background,
    rot15_img,
    rot15_mask,
    x,
    y
)

print("✓ Component pasted onto PCB")

save_result(
    result_rot15,
    "pcb_04_diode_rotate_15.png",
    "rotate",
    "+15 degrees",
    x,
    y
)


# --------------------------------------------------
# 8. SAVE METADATA
# --------------------------------------------------

print("\n[STEP 8/10] Saving augmentation metadata...")


metadata_header = [
    "original_image",
    "augmented_image",
    "component",
    "augmentation_type",
    "parameter",
    "old_x",
    "old_y",
    "new_x",
    "new_y",
    "width",
    "height"
]


existing_rows = []

# Keep metadata from IC/resistor/etc.
# Remove only old diode rows for pcb_01
# so rerunning does not duplicate them.
if metadata_path.exists():

    with open(
        metadata_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if not (
                row["original_image"] == image_path.name
                and row["component"] == "diode"
            ):

                existing_rows.append([
                    row["original_image"],
                    row["augmented_image"],
                    row["component"],
                    row["augmentation_type"],
                    row["parameter"],
                    row["old_x"],
                    row["old_y"],
                    row["new_x"],
                    row["new_y"],
                    row["width"],
                    row["height"]
                ])


with open(
    metadata_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        metadata_header
    )

    writer.writerows(
        existing_rows
    )

    writer.writerows(
        metadata_rows
    )


print("✓ Metadata saved")
print("  Location:", metadata_path)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n[STEP 9/10] Opening preview windows...")


cv2.imshow(
    "Original PCB",
    image
)

cv2.imshow(
    "1 - Diode Rotate 5 degrees",
    result_rot5
)

cv2.imshow(
    "2 - Diode Horizontal Flip",
    result_flip
)

cv2.imshow(
    "3 - Diode Move 10px",
    result_move10
)

cv2.imshow(
    "4 - Diode Move 25px",
    result_move25
)

cv2.imshow(
    "5 - Diode Rotate 15 degrees",
    result_rot15
)


print("\n======================================")
print("       WAITING FOR USER")
print("======================================")

print("Click on ANY OpenCV image window.")
print("Press ANY KEY to close all windows.")
print("The program is NOT frozen.\n")


cv2.waitKey(0)

print("✓ Key detected")
print("Closing preview windows...")

cv2.destroyAllWindows()


# --------------------------------------------------
# 10. FINISHED
# --------------------------------------------------

print("\n[STEP 10/10] Finalizing...")

print("\n======================================")
print("    DIODE AUGMENTATION COMPLETED")
print("======================================")

print("\nGenerated 5 augmented PCB images:")

print("  1. Diode Rotate +5°")
print("  2. Diode Horizontal Flip")
print("  3. Diode Move +10 px")
print("  4. Diode Move +25 px")
print("  5. Diode Rotate +15°")

print("\nOutput folder:")
print(output_folder)

print("\nMetadata file:")
print(metadata_path)

print("\n✓ PROGRAM FINISHED SUCCESSFULLY\n")