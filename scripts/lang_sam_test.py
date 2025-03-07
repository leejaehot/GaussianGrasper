from PIL import Image
from lang_sam import LangSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from lang_sam_visualier import visualize_results

if __name__ == "__main__":
    pwd_path = os.getcwd()
    model = LangSAM()
    image_path = "/home/shlee/ws_jclee/scene_0001_jclee/images/images_0001.png"
    image_pil = Image.open(image_path).convert("RGB")
    text_prompt = "plate."
    results = model.predict([image_pil], [text_prompt])
    visualize_results(image_path, results=results,
                      output_dir=pwd_path + "/vis_output/",
                      output_text_path=pwd_path + "/vis_output/output_results.txt")






