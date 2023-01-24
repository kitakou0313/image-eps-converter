import os
from PIL import Image
import glob

INPUTS_IMAGE_DIR = "./datas/inputs"
OUTPUTS_IMAGE_DIR = "./datas/outputs"

if __name__ == "__main__":
    input_image_paths = glob.glob(INPUTS_IMAGE_DIR+"/*")

    for image_path in input_image_paths:
        parent_dir, file_name = os.path.split(image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')

        file_name_without_extention = file_name.split(".")[0]
        image.save(f"{OUTPUTS_IMAGE_DIR}/{file_name_without_extention}.eps")
