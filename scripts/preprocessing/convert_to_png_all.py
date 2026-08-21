from PIL import Image
import pillow_avif
import os

input_folder = "data/originals"
output_folder = "data/originals"

os.makedirs(output_folder, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".avif")

files = [
    file for file in os.listdir(input_folder)
    if file.lower().endswith(valid_extensions)
]

for i, filename in enumerate(files, start=1):

    input_path = os.path.join(input_folder, filename)

    img = Image.open(input_path)
    img = img.convert("RGB")

    output_name = f"pcb_{i:02d}.png"
    output_path = os.path.join(output_folder, output_name)

    img.save(output_path)

    print(f"{filename}  ->  {output_name}")