import mlflow
import optuna
import training
import pickle
import os
from argparser import parse_arguments
from optuna import Trial
from optuna.study import Study


def objective(trial: Trial) -> float:
    """
    Optuna objective function that suggests hyperparameters, runs the training, and logs the result to MLflow.

    Args:
        trial (Trial): The Optuna trial object.

    Returns:
        float: The objective function value (validation loss).
    """
    default_args = parse_arguments()

    # Suggest and Update the default hyperparameters with the suggested hyperparameters
    # default_args.epochs = trial.suggest_int("epochs", 80, 140, step=20)
    default_args.bs = trial.suggest_int("bs", 50, 70)
    default_args.lr = trial.suggest_float("lr", 1e-4, 5e-3, step=2e-4)
    default_args.dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.2)
    default_args.feature_vector_size = trial.suggest_int(
        "feature_vector_size", 100, 300, step=50
    )

    # Run the training
    train_result = training.main(default_args)

    # Log the result to MLflow
    mlflow.log_metric("best_val_loss", train_result)

    return train_result


def save_study(study: Study, trial: Trial, study_name: str) -> None:
    """
    Saves the study to a file after each trial.

    Args:
        study (Study): The Optuna study object.
        trial (Trial): The Optuna trial object.
        study_name (str): The name of the study to save.
    """
    study_file = f"{study_name}.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study after trial {trial.number}")


def main():

    N_TRIALS = 2

    study_name = "RevEngineDev_optuna_study"

    # make a new folder for the study
    study_folder = "optuna_study_files"
    os.makedirs(study_folder, exist_ok=True)

    storage_db = f"""sqlite:///{os.path.join(study_folder, f"{study_name}.db")}"""
    study_file = os.path.join(study_folder, f"{study_name}.pkl")

    # Check if a saved study exists
    try:
        with open(study_file, "rb") as f:
            study = pickle.load(f)
    except FileNotFoundError:
        # If not, create a new study and save it
        study = optuna.create_study(
            direction="minimize", study_name=study_name, storage=storage_db
        )
        with open(study_file, "wb") as f:
            pickle.dump(study, f)

    # Optimize the study with the objective function and the custom callback
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        callbacks=[lambda study, trial: save_study(study, trial, study_name)],
    )

    # Save the updated study
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    # Save the best parameters
    best_params = f"""Best Trial: \n
    \t Value:  {study.best_trial.value} \n
    \t Params: {study.best_trial.params} \n
    """
    print(best_params, file=open("best_params.txt", "w"))


if __name__ == "__main__":
    main()
