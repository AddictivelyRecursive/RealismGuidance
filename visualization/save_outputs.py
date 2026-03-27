import csv
import os
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import custom_to_pil


def save_output_image(init_image_pil, target_image_pil, img, output_path):
    headings = ["Target Image", "Source Image", "Image"]
    font = ImageFont.load_default()

    max_height = max(init_image_pil.height, target_image_pil.height, img.height)
    total_width = init_image_pil.width + target_image_pil.width + img.width

    bbox = font.getbbox(headings[0])
    text_height = bbox[3] - bbox[1]

    output_image = Image.new("RGB", (total_width, max_height + text_height), "white")
    draw = ImageDraw.Draw(output_image)

    def add_image_with_heading(image, heading, x_position):
        bbox = font.getbbox(heading)
        heading_height = bbox[3] - bbox[1]
        draw.text((x_position, 0), heading, font=font, fill="black")
        output_image.paste(image, (x_position, heading_height))
        return x_position + image.width

    current_x = 0
    current_x = add_image_with_heading(init_image_pil, headings[0], current_x)
    current_x = add_image_with_heading(target_image_pil, headings[1], current_x)
    add_image_with_heading(img, headings[2], current_x)

    output_image.save(output_path)


def save_logs(
    logs,
    path,
    n_saved=0,
    key="sample",
    np_path=None,
    csv_file=None,
    init_image_path=None,
    target_image_path=None,
):
    if init_image_path is None or target_image_path is None:
        raise ValueError("init_image_path and target_image_path are required for saving outputs")

    init_image_pil = Image.open(init_image_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    target_image_pil = Image.open(target_image_path).convert("RGB").resize((256, 256), Image.LANCZOS)

    if csv_file is not None:
        csv_exists = os.path.isfile(csv_file)
        csv_handle = open(csv_file, "a", newline="")
        writer = csv.writer(csv_handle)
        if not csv_exists:
            writer.writerow(["source_image", "target_image", "combined_image", "output_image"])

    batch = logs[key]
    for x in batch:
        img = custom_to_pil(x)
        imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
        combined_path = os.path.join(path, f"combined_{key}_{n_saved:06}.png")

        save_output_image(init_image_pil, target_image_pil, img, combined_path)
        img.save(imgpath)

        if csv_file is not None:
            writer.writerow([
                os.path.abspath(init_image_path),
                os.path.abspath(target_image_path),
                os.path.abspath(combined_path),
                os.path.abspath(imgpath),
            ])

        n_saved += 1

    if csv_file is not None:
        csv_handle.close()

    return n_saved