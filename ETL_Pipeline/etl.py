from utils import clear_console, extract_data, transform_data, load_data


def main():
    clear_console()
    dataset_dir = "/home/ibad/Desktop/RevSearch/StanfordCarsDataset"
    mat_file_path = "cars_annos.mat"

    mat_file, dataset_image_paths = extract_data(mat_file_path, dataset_dir)
    train, test, val = transform_data(
        mat_file, dataset_image_paths, dataset_dir, minimum_resolution=(20, 20)
    )

    # Lengths of the dataframes
    print("Train: ", len(train))
    print("Test: ", len(test))
    print("Val: ", len(val))

    # print number of examples per class
    print("Train: ")
    print(train["class"].value_counts())
    print("Test: ")
    print(test["class"].value_counts())
    print("Val: ")
    print(val["class"].value_counts())

    # print unique labels
    print("Train: ")
    print(sorted((train["class"].unique())))

    load_data(train, test, val, dataset_dir)


if __name__ == "__main__":
    main()
