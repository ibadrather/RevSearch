"""
The DeepImageSearch library allows you to create an image search engine with your custom-trained model. 
You can follow these steps to integrate your custom model into DeepImageSearch:

Install the DeepImageSearch library:
pip install deepimagesearch
"""
import os
import sys

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# print the sys.path
print(sys.path)

from DeepImageSearch.DeepImageSearch import Load_Data, Search_Setup
from preprocessing_fn import preprocess_fn
from inference_utils import infer_onnx_model, load_onnx_model
from typing import List

# list all the images in the directory and subdirectories
def list_images(path: str) -> List[str]:
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                images.append(os.path.join(root, file))
    return images

image_list_all = list_images("Car196_Combined/images")
image_list = list_images("Car196_Combined/images")[:100]

print("Number of images: ", len(image_list))

onnx_model = load_onnx_model("RevSearchEngine/pretrained_models/efficientnet_2023_05_03-21_42_46/best_encoder_efficientnet.onnx")

# Setup search engine
search_engine = Search_Setup(
    image_list=image_list,
    custom_model_name="efficientnet-onnx",
    preprocess_fn=preprocess_fn,
    model=onnx_model,
)

# Get similar images
similar_images = search_engine.get_similar_images(
    image_path=image_list_all[0], 
    number_of_images=9,
)

# Plot similar images
print(similar_images)