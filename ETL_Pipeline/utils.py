import os
from typing import Tuple, List
import pandas as pd
import scipy.io
from tqdm import tqdm
from natsort import natsorted
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


def clear_console() -> None:
    """Clear the console."""
    os.system("clear")


def load_mat_file(file_path: str) -> dict:
    """Load a .mat file and return its content as a dictionary.

    Args:
        file_path: The path of the .mat file.

    Returns:
        A dictionary containing the content of the .mat file.
    """
    return scipy.io.loadmat(file_path)


def clean_image_names(column: pd.Series) -> pd.Series:
    """Clean the image names in a pandas Series.

    Args:
        column: A pandas Series containing image paths.

    Returns:
        A pandas Series containing the cleaned image names.
    """
    return column.apply(lambda x: x[0]).apply(lambda x: x.split("/")[-1][1:])


def clean_column(column: pd.Series, index: int = 0) -> pd.Series:
    """Clean the elements of a pandas Series.

    Args:
        column: A pandas Series containing elements to be cleaned.
        index: An integer index value used to clean the elements.

    Returns:
        A pandas Series containing the cleaned elements.
    """
    return column.apply(lambda x: x[index][index])


def create_car_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a car dataset from a given DataFrame.

    Args:
        df: A DataFrame containing car data.

    Returns:
        A DataFrame containing the car dataset with cleaned columns.
    """
    image_names = clean_image_names(df["relative_im_path"])
    bbox_x1 = clean_column(df["bbox_x1"])
    bbox_y1 = clean_column(df["bbox_y1"])
    bbox_x2 = clean_column(df["bbox_x2"])
    bbox_y2 = clean_column(df["bbox_y2"])
    class_ = clean_column(df["class"])
    test = clean_column(df["test"])

    return pd.DataFrame(
        {
            "image_names": image_names,
            "bbox_x1": bbox_x1,
            "bbox_y1": bbox_y1,
            "bbox_x2": bbox_x2,
            "bbox_y2": bbox_y2,
            "class": class_,
            "test": test,
        }
    )


def list_files_recursive(directory: str, file_format: str) -> List[str]:
    """List all files with a specific format in a directory, recursively.

    Args:
        directory: The directory to search for files.
        file_format: The file format to look for.

    Returns:
        A list of file paths with the specified format.
    """
    file_paths = []

    for root, _, files in os.walk(directory):
        files = natsorted(files)
        for file in files:
            if file.endswith(file_format):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                file_paths.append(rel_path)

    return file_paths


def create_rev_search_df(
    car_dataset: pd.DataFrame,
    dataset_dir: str,
    dataset_image_paths: List[str],
    minimum_resolution: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Create a reverse search DataFrame from the car dataset and image paths.

    Args:
        car_dataset: A DataFrame containing car data.
        dataset_dir: The directory containing the dataset images.
        dataset_image_paths: A list of dataset image paths.

    Returns:
        A DataFrame containing image paths and classes for reverse search.
    """
    train_image_paths = [path for path in dataset_image_paths if "train" in path]
    test_image_paths = [path for path in dataset_image_paths if "test" in path]

    train_image_name_to_path = {
        os.path.basename(path): path for path in train_image_paths
    }
    test_image_name_to_path = {
        os.path.basename(path): path for path in test_image_paths
    }

    rev_search_df = pd.DataFrame(columns=["image_paths", "class"])

    for image_name, class_, test in tqdm(
        car_dataset[["image_names", "class", "test"]].values
    ):
        if test == 0 and image_name in train_image_name_to_path:
            dataset_image_path = train_image_name_to_path[image_name]
        elif test == 1 and image_name in test_image_name_to_path:
            dataset_image_path = test_image_name_to_path[image_name]
        else:
            continue

        image_path = os.path.join(dataset_dir, dataset_image_path)

        # Check if the image exists and has a minimum resolution bigger that given
        if not os.path.exists(image_path):
            continue
        else:
            image = Image.open(image_path)
            if (
                image.size[0] < minimum_resolution[0]
                or image.size[1] < minimum_resolution[1]
            ):
                continue

        rev_search_df_row = pd.DataFrame(
            [[image_path, class_]], columns=["image_paths", "class"]
        )
        rev_search_df = pd.concat([rev_search_df, rev_search_df_row], ignore_index=True)

    return rev_search_df


def train_test_val_split_stratified_80_10_10(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the input DataFrame into stratified train, test, and validation sets.

    Args:
        df: The input DataFrame to split.
        random_state: The random state to use for reproducibility.

    Returns:
        A tuple containing the train, test, and validation DataFrames.
    """
    train, test = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df["class"]
    )

    test, val = train_test_split(
        test, test_size=0.5, random_state=random_state, stratify=test["class"]
    )

    return train, test, val


def save_csvs(
    train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame, dataset_dir: str
) -> None:
    """Save the train, test, and validation DataFrames as CSV files in the dataset directory.

    Args:
        train: The train DataFrame.
        test: The test DataFrame.
        val: The validation DataFrame.
        dataset_dir: The directory to save the CSV files.
    """
    train.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)
    val.to_csv(os.path.join(dataset_dir, "val.csv"), index=False)


def extract_data(mat_file_name: str, dataset_dir: str) -> Tuple[dict, List[str]]:
    """Load the MAT file and dataset image paths.

    Args:
        mat_file_path: The path to the MAT file.
        dataset_dir: The directory containing the dataset images.

    Returns:
        A tuple containing the MAT file content and a list of dataset image paths.
    """
    mat_file_path = os.path.join(dataset_dir, mat_file_name)
    mat_file = load_mat_file(mat_file_path)
    dataset_image_paths = list_files_recursive(dataset_dir, ".jpg")
    return mat_file, dataset_image_paths


def transform_data(
    mat_file: dict,
    dataset_image_paths: List[str],
    dataset_dir: str,
    minimum_resolution: Tuple[int, int] = (224, 224),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process the data and split it into stratified train, test, and validation sets.

    Args:
        mat_file: The MAT file content.
        dataset_image_paths: A list of dataset image paths.
        dataset_dir: The directory containing the dataset images.

    Returns:
        A tuple containing the train, test, and validation DataFrames.
    """
    dataset_dir = "StanfordCarsDataset"

    df = pd.DataFrame(mat_file["annotations"][0])
    car_dataset = create_car_dataset(df)
    rev_search_df = create_rev_search_df(
        car_dataset,
        dataset_dir,
        dataset_image_paths,
        minimum_resolution=minimum_resolution,
    )
    train, test, val = train_test_val_split_stratified_80_10_10(rev_search_df)
    return train, test, val


def load_data(
    train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame, dataset_dir: str
) -> None:
    """Save the train, test, and validation sets as CSV files in the dataset directory.

    Args:
        train: The train DataFrame.
        test: The test DataFrame.
        val: The validation DataFrame.
        dataset_dir: The directory to save the CSV files.
    """
    save_csvs(train, test, val, dataset_dir)


import pandas as pd
from typing import List


def convert_one_hot_to_labels(
    input_file: str, filename_col: str, class_label_col: str
) -> pd.DataFrame:
    """
    Convert a one-hot encoded CSV dataset to a dataset with class labels.

    Args:
        input_file (str): Path to the input CSV file with one-hot encoded data.
        filename_col (str): Name of the column containing the filenames.
        class_label_col (str): Name of the column to store the class labels.

    Returns:
        pd.DataFrame: A DataFrame containing the filename and class label columns.
    """
    # Read the input CSV file
    data = pd.read_csv(input_file)

    # # Ensure that all columns except the filename column are numeric
    # for col in data.columns:
    #     if col != filename_col:
    #         data[col] = pd.to_numeric(data[col], errors='coerce')

    # Initialize an empty list to store the class labels
    class_labels: List[str] = []

    # Iterate over each row in the dataset
    for index, row in data.iterrows():
        # Find the index of the first occurrence of 1 in the row (ignoring the filename column)
        # Find the index of the first occurrence of 1 in the row (ignoring the filename column)
        class_label = row.drop(filename_col).index[np.argmax(row.drop(filename_col))]

        class_labels.append(class_label)

    # Add a new column with the class labels
    data[class_label_col] = class_labels

    # Keep only the filename and class label columns
    data = data[[filename_col, class_label_col]]

    return data
