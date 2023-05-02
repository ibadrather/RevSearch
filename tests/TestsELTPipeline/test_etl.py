# from unittest.mock import MagicMock, patch
# import pandas as pd
# import os
# import sys

# # Get the absolute path of the parent directory of the script
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# # Add the parent directory to sys.path
# sys.path.append(parent_dir)


# from ETL_Pipeline.utils import (
#     load_mat_file,
#     create_car_dataset,
#     train_test_val_split_stratified_80_10_10,
#     extract_data,
#     load_data,
#     transform_data,
# )


# def test_load_mat_file():
#     with patch("scipy.io.loadmat", MagicMock(return_value={"key": "value"})):
#         mat_content = load_mat_file("path/to/file.mat")
#         assert mat_content == {"key": "value"}


# def test_create_car_dataset():
#     raw_df = pd.DataFrame(
#         {
#             "relative_im_path": [["/somepath/image1.jpg"]],
#             "bbox_x1": [[0]],
#             "bbox_y1": [[0]],
#             "bbox_x2": [[100]],
#             "bbox_y2": [[100]],
#             "class": [[1]],
#             "test": [[0]],
#         }
#     )
#     result = create_car_dataset(raw_df)
#     expected = pd.DataFrame(
#         {
#             "image_names": ["image1.jpg"],
#             "bbox_x1": [0],
#             "bbox_y1": [0],
#             "bbox_x2": [100],
#             "bbox_y2": [100],
#             "class": [1],
#             "test": [0],
#         }
#     )
#     assert result.equals(expected)


# def test_train_test_val_split_stratified_80_10_10():
#     df = pd.DataFrame({"class": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
#     train, test, val = train_test_val_split_stratified_80_10_10(df)

#     assert len(train) == 8
#     assert len(test) == 1
#     assert len(val) == 1
#     assert train["class"].value_counts().tolist() == [2, 2, 2, 2, 2]
#     assert test["class"].value_counts().tolist() == [1]
#     assert val["class"].value_counts().tolist() == [1]


# def test_extract_data():
#     with patch("your_module.load_mat_file", MagicMock(return_value={"key": "value"})):
#         with patch(
#             "your_module.list_files_recursive",
#             MagicMock(return_value=["path1", "path2"]),
#         ):
#             mat_content, dataset_image_paths = extract_data(
#                 "mat_file_name", "dataset_dir"
#             )
#             assert mat_content == {"key": "value"}
#             assert dataset_image_paths == ["path1", "path2"]


# def test_transform_data():
#     mat_file = {
#         "annotations": [
#             [
#                 {
#                     "relative_im_path": ["/somepath/image1.jpg"],
#                     "bbox_x1": [0],
#                     "bbox_y1": [0],
#                     "bbox_x2": [100],
#                     "bbox_y2": [100],
#                     "class": [1],
#                     "test": [0],
#                 }
#             ]
#         ]
#     }
#     dataset_image_paths = ["path1", "path2"]
#     dataset_dir = "StanfordCarsDataset"

#     with patch("your_module.create_car_dataset", MagicMock(return_value=pd.DataFrame())):
#         with patch("your_module.create_rev_search_df", MagicMock(return_value=pd.DataFrame())):
#             with patch("your_module.train_test_val_split_stratified_80_10_10", MagicMock(return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))):
#                 train, test, val = transform_data(mat_file, dataset_image_paths, dataset_dir)
#                 assert isinstance(train, pd.DataFrame)
#                 assert isinstance(test, pd.DataFrame)
#                 assert isinstance(val, pd.DataFrame)


# def test_load_data():
#     train = pd.DataFrame({"class": [1, 1, 2, 2, 3, 3, 4, 4]})
#     test = pd.DataFrame({"class": [1, 2]})
#     val = pd.DataFrame({"class": [1, 2]})
#     dataset_dir = "StanfordCarsDataset"

#     with patch("your_module.save_csvs", MagicMock()) as mock_save_csvs:
#         load_data(train, test, val, dataset_dir)
#         mock_save_csvs.assert_called_once_with(train, test, val, dataset_dir)
