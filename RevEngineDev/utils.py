import mlflow
import argparse
import pandas as pd
from sklearn.utils import resample


def log_arguments_to_mlflow(args: argparse.Namespace) -> None:
    """
    Log the command-line arguments to MLflow.

    Args:
        args: argparse.Namespace object containing the parsed command-line arguments
    """
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)


def balance_dataframe_by_resampling_class(
    dataframe: pd.DataFrame,
    class_column: str,
    sample_size: int,
    seed: int = 1234,
    replace: bool = True,
) -> pd.DataFrame:
    """
    Balances a given dataframe by resampling all classes to the specified sample size.

    Args:
        dataframe (pd.DataFrame): The input imbalanced dataframe.
        class_column (str): The column name that contains the class labels.
        sample_size (int): The number of samples for each class after balancing.
        seed (int, optional): The random state for reproducibility. Defaults to 1234.
        replace (bool, optional): Whether to perform oversampling (True) or undersampling (False). Defaults to True.

    Returns:
        pd.DataFrame: The balanced dataframe.
    """
    # Find unique classes
    unique_classes = dataframe[class_column].unique()
    separated_dataframes = []

    # Separate all classes
    for class_ in unique_classes:
        separated_dataframes.append(dataframe[dataframe[class_column] == class_])

    # Resample all classes based on sample size
    resampled_dataframes = []
    for class_dataframe in separated_dataframes:
        resampled_dataframes.append(
            resample(
                class_dataframe,
                replace=replace,
                n_samples=sample_size,
                random_state=seed,
            )
        )

    # Combine all classes
    balanced_dataframe = pd.concat(resampled_dataframes)

    return balanced_dataframe
