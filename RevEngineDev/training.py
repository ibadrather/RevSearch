import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional
from argparser import parse_arguments
from utils import log_arguments_to_mlflow

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from augmentations import AlbumentationsDataAugmentation
from trainer import train
from model import RevSearchFeatureExtractor
from dataloading import StanfordCarDataset

import mlflow

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/ibad/Desktop/RevSearch/Car196_Combined/images")
DATASET_DIR = Path("/home/ibad/Desktop/RevSearch/Car196_Combined/images")

def main(arg_namespace: Optional[argparse.Namespace] = None) -> float:
    # Clear console
    os.system("clear")

    # Parse arguments
    args = parse_arguments() if arg_namespace is None else arg_namespace

    TIMESTAMP = time.strftime("%Y_%m_%d-%H_%M_%S")
    model_output_dir = Path("RevEngineDev_Models") / f"{args.arch}_{TIMESTAMP}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache and set random seeds for reproducibility
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # Data augmentation
    augmentations = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotate_90": True,
        "transpose": True,
        "medium_augmentations": True,
        "clahe": True,
        "random_brightness_contrast": True,
        "random_gamma": True,
    }
    data_augmentation = AlbumentationsDataAugmentation(
        image_size=args.image_size, options=augmentations
    )
    data_augmentation = None

    train_dataset = StanfordCarDataset(
            csv_file=DATA_DIR / "train.csv",
            dataset_dir=DATASET_DIR,
            transforms=data_augmentation,
        )

    val_dataset = StanfordCarDataset(
        csv_file=DATA_DIR / "val.csv",
        dataset_dir=DATASET_DIR,
        transforms=data_augmentation,
    )

    # Initialize the model
    model = RevSearchFeatureExtractor(
        num_classes=train_dataset.num_classes,
        dropout=args.dropout,
        feature_vector_size=args.feature_vector_size,
    )

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set up the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Set the device to be used for training
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Close any older mlflow runs
    mlflow.end_run()

    mlflow.set_experiment("RevEngineDev_Experiment")
    with mlflow.start_run():
        # Setup datasets

        # Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

        # Log the arguments to MLFlow
        log_arguments_to_mlflow(args)

        # Train the model
        train_loss, train_acc, val_loss, val_acc, best_val_loss = train(
                        model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            save_dir=model_output_dir,
            verbose=True,
        )

        # Make a logs directory inside the model_output_dir and save the logs there
        logs_dir = model_output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Save the logs
        logs_df = pd.DataFrame(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        logs_df.to_csv(logs_dir / "logs.csv", index=False)

        # Plot the loss and accuracy curves
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(logs_dir / "loss.png")

        plt.figure(figsize=(10, 7))
        plt.plot(train_acc, label="train_acc")
        plt.plot(val_acc, label="val_acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(logs_dir / "accuracy.png")

        # Mlflow artifact the plots
        mlflow.log_artifact(logs_dir / "loss.png")
        mlflow.log_artifact(logs_dir / "accuracy.png")

    return best_val_loss


if __name__ == "__main__":
    main()
