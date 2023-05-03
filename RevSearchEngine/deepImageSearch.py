"""
The DeepImageSearch library allows you to create an image search engine with your custom-trained model. 
You can follow these steps to integrate your custom model into DeepImageSearch:

Install the DeepImageSearch library:
pip install deepimagesearch
"""

from DeepImageSearch import ImageSearchModel, ImageSearch
from preprocessing_fn import preprocess_fn
from inference_utils import infer_onnx_model, load_onnx_model
import os
from typing import List

# Create a custom ImageSearchModel class that inherits from the base ImageSearchModel class.
# Override the feature_extractor method to use your custom-trained model for generating feature vectors:


class RevSearchEngineModel(ImageSearchModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.session, self.input_name, self.output_name = load_onnx_model(model_path)

    def feature_extractor(self, image) -> List[float]:
        # Preprocess the image as required by your custom model
        image = preprocess_fn(image, (224, 224))

        # Generate feature vector using your custom model
        feature_vector = self.session.run([self.output_name], {self.input_name: image})[
            0
        ]

        return feature_vector


# Initialize your custom model and the CustomImageSearchModel instance:

rev_search_engine_model = RevSearchEngineModel(
    model_path="RevSearchEngine/pretrained_models/efficientnet_2023_05_03-16_27_49/best_encoder_efficientnet.onnx"
)


# Index your images:

# Assuming you have a list of image paths
image_paths = os.listdir("Car196_Combined/images")[:20]

for image_path in image_paths:
    rev_search_engine_model.add_image(image_path)

# Save the indexed images and features:
rev_search_engine_model.save_index("index.pkl")

# Load the index when needed:
rev_search_engine_model.load_index("index.pkl")

# Perform reverse image search:
top_k_results = 10

similar_images = rev_search_engine_model.search_image(
    image_paths[99], num_results=top_k_results
)

# Display or return the search results as desired.
# Now, you've integrated your custom-trained model into DeepImageSearch and can use it for reverse image search tasks.

print(similar_images)
