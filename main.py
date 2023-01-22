import os
from PIL import Image
import glob

INPUTS_IMAGE_DIR = "./inputs"
OUTPUTS_IMAGE_DIR = "./outputs"

if __name__ == "__main__":
    input_image_paths = glob.glob(INPUTS_IMAGE_DIR+"/*")

    for image_path in input_image_paths:
        file_name = os.path.split(image_path)[1]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image.save(OUTPUTS_IMAGE_DIR+"/"+file_name.split(".")[0]+".eps")
