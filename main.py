import os
from PIL import Image
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NTGA attack!")
    parser.add_argument("--inputs_image_dir",
                        default="./datas/inputs", type=str)
    parser.add_argument("--outputs_image_dir", type=str,
                        default="./datas/outputs")

    args = parser.parse_args()

    INPUTS_IMAGE_DIR = args.inputs_image_dir
    OUTPUTS_IMAGE_DIR = args.outputs_image_dir

    input_image_paths = glob.glob(INPUTS_IMAGE_DIR+"/*")

    for image_path in input_image_paths:
        parent_dir, file_name = os.path.split(image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')

        file_name_without_extention = file_name.split(".")[0]
        image.save(f"{OUTPUTS_IMAGE_DIR}/{file_name_without_extention}.eps")
