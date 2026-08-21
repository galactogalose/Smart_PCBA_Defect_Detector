from pathlib import Path
import cv2
import csv
import numpy as np


print("\n======================================")
print("  COMPONENT AUGMENTATION DISPLAY")
print("======================================\n")


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

PCB_NAME = "pcb_04"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

base_folder = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / PCB_NAME
)

csv_path = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotations.csv"
)


# --------------------------------------------------
# DISPLAY SETTINGS
# --------------------------------------------------

# White space between augmentation images
HORIZONTAL_GAP = 50

# Height of augmentation title area
TITLE_HEIGHT = 90

# Extra padding around original bounding box
# Makes the green box easier to see
BOX_PADDING = 20

# Green bounding box thickness
BOX_THICKNESS = 8


# --------------------------------------------------
# 2. COMPONENTS
# --------------------------------------------------

components = [
    ("IC", "IC"),
    ("CAPACITOR", "capacitor"),
    ("DIODE", "diode")
]


# --------------------------------------------------
# 3. AUGMENTATION ORDER
# --------------------------------------------------

augmentations = [
    ("Rotate +5°", "rotate_05"),
    ("Horizontal Flip", "flip_horizontal"),
    ("Move +10 px", "move_10px"),
    ("Move +25 px", "move_25px"),
    ("Rotate +15°", "rotate_15")
]


# --------------------------------------------------
# 4. READ BOUNDING BOXES FROM CSV
# --------------------------------------------------

print("[1/6] Reading annotations...")

component_boxes = {}


with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["image"] != f"{PCB_NAME}.png":
            continue

        component = row["component"]

        if component not in [
            "IC",
            "capacitor",
            "diode"
        ]:
            continue

        # Currently take first annotation
        # for each component type
        if component not in component_boxes:

            component_boxes[component] = (
                int(row["x"]),
                int(row["y"]),
                int(row["width"]),
                int(row["height"])
            )


for component, box in component_boxes.items():

    print(
        f"✓ {component}: {box}"
    )


# --------------------------------------------------
# 5. LOAD AUGMENTED IMAGE
# --------------------------------------------------

def load_augmented_image(
    component_folder,
    component_name,
    suffix
):

    filename = (
        f"{PCB_NAME}_"
        f"{component_name}_"
        f"{suffix}.png"
    )

    image_path = (
        base_folder
        / component_folder
        / filename
    )

    print("  Loading:", filename)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load:\n{image_path}"
        )

    return image


# --------------------------------------------------
# 6. DRAW LARGE GREEN BOX
# --------------------------------------------------

def draw_component_box(
    image,
    component_name,
    suffix
):

    output = image.copy()


    if component_name not in component_boxes:

        print(
            f"WARNING: No bbox found for {component_name}"
        )

        return output


    x, y, w, h = component_boxes[
        component_name
    ]


    # --------------------------------------------------
    # Account for movement augmentation
    # --------------------------------------------------

    new_x = x
    new_y = y


    if suffix == "move_10px":

        new_x = x + 10


    elif suffix == "move_25px":

        new_x = x + 25


    # --------------------------------------------------
    # Make bbox larger
    # --------------------------------------------------

    x1 = max(
        0,
        new_x - BOX_PADDING
    )

    y1 = max(
        0,
        new_y - BOX_PADDING
    )

    x2 = min(
        output.shape[1] - 1,
        new_x + w + BOX_PADDING
    )

    y2 = min(
        output.shape[0] - 1,
        new_y + h + BOX_PADDING
    )


    # --------------------------------------------------
    # Draw GREEN bounding box
    # --------------------------------------------------

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        BOX_THICKNESS
    )


    return output


# --------------------------------------------------
# 7. ADD AUGMENTATION TITLE
# --------------------------------------------------

def add_title(
    image,
    title
):

    # Add white panel above image
    output = cv2.copyMakeBorder(
        image,
        TITLE_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 1.25

    thickness = 3


    text_size, _ = cv2.getTextSize(
        title,
        font,
        font_scale,
        thickness
    )


    text_width = text_size[0]


    text_x = max(
        10,
        (image.shape[1] - text_width) // 2
    )


    text_y = 58


    cv2.putText(
        output,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


    return output


# --------------------------------------------------
# 8. CREATE WHITE GAP
# --------------------------------------------------

def create_gap(height):

    return np.full(
        (
            height,
            HORIZONTAL_GAP,
            3
        ),
        255,
        dtype=np.uint8
    )


# --------------------------------------------------
# 9. CREATE ONE COMPONENT DISPLAY
# --------------------------------------------------

def create_component_display(
    display_name,
    component_name
):

    print("\n======================================")
    print(f" BUILDING {display_name} DISPLAY")
    print("======================================")

    images = []

    # --------------------------------------------------
    # LOAD + BOX + TITLE ALL 5 AUGMENTATIONS
    # --------------------------------------------------

    for title, suffix in augmentations:

        image = load_augmented_image(
            component_name,
            component_name,
            suffix
        )

        # Draw larger green bounding box
        image = draw_component_box(
            image,
            component_name,
            suffix
        )

        # Add augmentation title
        image = add_title(
            image,
            title
        )

        images.append(image)


    # --------------------------------------------------
    # 3 IMAGES IN FIRST ROW
    # --------------------------------------------------

    row1_parts = []

    for i in range(3):

        row1_parts.append(
            images[i]
        )

        # Add gap only between images
        if i < 2:

            row1_parts.append(
                create_gap(
                    images[i].shape[0]
                )
            )


    row1 = np.hstack(
        row1_parts
    )


    # --------------------------------------------------
    # 2 IMAGES IN SECOND ROW
    # --------------------------------------------------

    row2_parts = []

    for i in range(3, 5):

        row2_parts.append(
            images[i]
        )

        if i < 4:

            row2_parts.append(
                create_gap(
                    images[i].shape[0]
                )
            )


    row2 = np.hstack(
        row2_parts
    )


    # --------------------------------------------------
    # CENTER SECOND ROW
    # --------------------------------------------------

    # Row 1 is wider because it contains 3 PCBs.
    # Add white padding equally on left/right of row 2.

    width_difference = (
        row1.shape[1]
        - row2.shape[1]
    )

    left_padding = (
        width_difference // 2
    )

    right_padding = (
        width_difference
        - left_padding
    )


    row2 = cv2.copyMakeBorder(
        row2,
        0,
        0,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


    # --------------------------------------------------
    # GAP BETWEEN ROW 1 AND ROW 2
    # --------------------------------------------------

    VERTICAL_GAP = 50

    vertical_gap = np.full(
        (
            VERTICAL_GAP,
            row1.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )


    # --------------------------------------------------
    # COMBINE BOTH ROWS
    # --------------------------------------------------

    final_display = np.vstack([
        row1,
        vertical_gap,
        row2
    ])


    # --------------------------------------------------
    # SAVE HIGH-RESOLUTION IMAGE
    # --------------------------------------------------

    output_path = (
        base_folder
        / f"{PCB_NAME}_{component_name}_augmentations.png"
    )


    success = cv2.imwrite(
        str(output_path),
        final_display
    )


    if success:

        print("\n✓ High-resolution display saved")

        print("  File:", output_path)

        print(
            "  Resolution:",
            final_display.shape[1],
            "x",
            final_display.shape[0]
        )

    else:

        print(
            "✗ ERROR saving:",
            output_path
        )


    return final_display

# --------------------------------------------------
# 10. GENERATE ALL THREE DISPLAYS
# --------------------------------------------------

print("\n[2/6] Creating component displays...")


ic_display = create_component_display(
    "IC",
    "IC"
)

capacitor_display = create_component_display(
    "CAPACITOR",
    "capacitor"
)

diode_display = create_component_display(
    "DIODE",
    "diode"
)


print("\n✓ All three component displays created")


# --------------------------------------------------
# 11. PREVIEW FUNCTION
# --------------------------------------------------

def create_preview(
    image,
    max_width=1400,
    max_height=760
):

    image_h, image_w = image.shape[:2]

    # Leave some room for Windows title bar,
    # VS Code, taskbar, etc.
    safe_width = int(
        max_width * 0.88
    )

    safe_height = int(
        max_height * 0.88
    )


    scale_w = (
        safe_width / image_w
    )

    scale_h = (
        safe_height / image_h
    )


    scale = min(
        scale_w,
        scale_h,
        1.0
    )


    preview_w = int(
        image_w * scale
    )

    preview_h = int(
        image_h * scale
    )


    preview = cv2.resize(
        image,
        (
            preview_w,
            preview_h
        ),
        interpolation=cv2.INTER_AREA
    )


    print(
        "Preview resolution:",
        preview_w,
        "x",
        preview_h
    )


    return preview


# --------------------------------------------------
# 12. DISPLAY EACH COMPONENT ONE BY ONE
# --------------------------------------------------

print("\n[5/6] Opening previews...")


# Store each display separately.
#
# This prevents accidentally showing
# ic_display three times.

displays = [
    (
        "IC",
        ic_display
    ),

    (
        "Capacitor",
        capacitor_display
    ),

    (
        "Diode",
        diode_display
    )
]


for index, (
    component_title,
    display_image
) in enumerate(displays):


    print("\n--------------------------------------")
    print(
        f"Opening {component_title} preview"
    )
    print("--------------------------------------")


    # Different window name every time
    window_name = (
        f"{PCB_NAME} - "
        f"{component_title} Augmentations"
    )


    # Generate preview from the CORRECT
    # component display
    preview = create_preview(
        display_image
    )


    print(
        "Displaying:",
        component_title
    )


    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    cv2.imshow(
        window_name,
        preview
    )


    cv2.resizeWindow(
        window_name,
        preview.shape[1],
        preview.shape[0]
    )


    if index < len(displays) - 1:

        print(
            "Press ANY KEY for next component."
        )

    else:

        print(
            "Press ANY KEY to finish."
        )


    cv2.waitKey(0)

    # Destroy THIS window before
    # displaying the next one
    cv2.destroyWindow(
        window_name
    )


cv2.destroyAllWindows()


# --------------------------------------------------
# 13. FINISHED
# --------------------------------------------------

print("\n[6/6] Finished.")


print("\n======================================")
print("      ALL DISPLAYS GENERATED")
print("======================================")


print("\nSaved files:")


ic_output = (
    base_folder
    / f"{PCB_NAME}_IC_augmentations.png"
)

capacitor_output = (
    base_folder
    / f"{PCB_NAME}_capacitor_augmentations.png"
)

diode_output = (
    base_folder
    / f"{PCB_NAME}_diode_augmentations.png"
)


print("\nIC:")
print(ic_output)

print("\nCapacitor:")
print(capacitor_output)

print("\nDiode:")
print(diode_output)


print("\n✓ PROGRAM FINISHED\n")