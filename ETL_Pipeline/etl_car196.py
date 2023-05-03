import os
from utils import clear_console, extract_data, transform_data, load_data, convert_one_hot_to_labels
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    os.system("clear")
    test_csv_path = "Car196/test/_classes.csv"
    train_csv_path = "Car196/train/_classes.csv"
    val_csv_path = "Car196/valid/_classes.csv"

    # Convert all from one-hot to labels
    train_df = convert_one_hot_to_labels(train_csv_path, "filename", "class")
    test_df = convert_one_hot_to_labels(test_csv_path, "filename", "class")
    val_df = convert_one_hot_to_labels(val_csv_path, "filename", "class")

    # Shape of the dataframes
    print("Train: ", train_df.shape)
    print("Test: ", test_df.shape)
    print("Val: ", val_df.shape)

    # Value counts of the dataframes classes
    print("Train: ", len(train_df["class"].value_counts()))
    print("Test: ", len(test_df["class"].value_counts()))
    print("Val: ", len(val_df["class"].value_counts()))


    # print(sorted(list(train_df["class"].value_counts().keys())))


    # combine all dataframes
    frames = [train_df, test_df, val_df]
    result = pd.concat(frames)

    # save the combined dataframe
    result.to_csv("Car196/all_data.csv", index=False)

    # drop classes with less than 10 images
    result = result.groupby("class").filter(lambda x: len(x) > 30)
    
    # train test srtatified split
    train, test = train_test_split(result, test_size=0.2, stratify=result["class"], random_state=42)
    test, val = train_test_split(test, test_size=0.5, stratify=test["class"], random_state=42)

    # save the train, test, val dataframes
    train.to_csv("Car196/train.csv", index=False)
    test.to_csv("Car196/test.csv", index=False)
    val.to_csv("Car196/val.csv", index=False)

    # Value counts of the dataframes classes
    print("Train: ", len(train["class"].value_counts()))
    print("Test: ", len(test["class"].value_counts()))
    print("Val: ", len(val["class"].value_counts()))

    # Shape of the dataframes
    print("Train: ", train.shape)
    print("Test: ", test.shape)
    print("Val: ", val.shape)





if __name__ == "__main__":
    main()
