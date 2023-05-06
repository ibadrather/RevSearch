import os


def image_data_with_features_pkl(metadata_dir, model_name):
    data_dir = os.path.join(metadata_dir, f"{model_name}")

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    image_data_with_features_pkl = os.path.join(data_dir, "image_data_features.pkl")
    return image_data_with_features_pkl


def image_features_vectors_idx(metadata_dir, model_name):
    data_dir = os.path.join(metadata_dir, f"{model_name}")

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    image_features_vectors_idx = os.path.join(data_dir, "image_features_vectors.idx")
    return image_features_vectors_idx
