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


from DeepImageSearch.DeepImageSearch import Load_Data, Search_Setup
from inference_utils import CustomFeatureExtractor

os.system("clear")


# Load images from a folder
image_list_all = Load_Data().from_folder(["Car196_Combined/images"])
image_list = image_list_all[:]

print("Number of images: ", len(image_list))

model_path = "RevSearchEngine/pretrained_models/efficientnet_2023_05_03-21_42_46/best_encoder_efficientnet.onnx"

feature_extractor = CustomFeatureExtractor(
    model_path=model_path,
)

# Setup search engine
search_engine = Search_Setup(
    image_list=image_list,
    custom_feature_extractor=feature_extractor,
    custom_feature_extractor_name="efficientnet_onnx",
    # image_count=100,
)

# Index the images
search_engine.run_index()

# Get metadata
metadata = search_engine.get_image_metadata_file()
# print(metadata.head())

# Get similar images
# similar_images = search_engine.get_similar_images(
#     image_path=image_list_all[3000],
#     number_of_images=9,
# )


image_path = "car.jpg"

# Plot similar images
# print(similar_images)
search_engine.plot_similar_images(image_path=image_path, number_of_images=5)
