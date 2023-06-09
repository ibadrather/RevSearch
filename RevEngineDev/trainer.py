from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
from tqdm import tqdm
import torch.nn.functional as F
import os.path as osp
from argparse import Namespace
import mlflow
import warnings
import torch.onnx


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args: argparse.ArgumentParser,
    save_dir: str = "",
    verbose: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    start_time = time.time()

    model.train()
    model.to(device)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_val_loss = 10e6

    for epoch in range(args.epochs):
        accuracy_one_epoch = 0
        loss_one_epoch = 0

        with tqdm(
            total=len(train_dataloader),
            leave=True,
            desc=f"Epoch [{epoch+1}/{args.epochs}]",
        ) as loop:
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                preds = F.log_softmax(logits, dim=1).argmax(dim=1)
                preds = preds.cpu().numpy().flatten()
                labels = labels.cpu().numpy().flatten()

                accuracy_one_epoch += (preds == labels).sum() / len(labels)
                loss_one_epoch += loss.item()

                loop.update(1)
                loop.set_postfix(
                    train_loss=loss_one_epoch / (i + 1),
                    t_acc=accuracy_one_epoch / (i + 1),
                    elapsed=(time.time() - start_time) / 60,
                )

            train_loss.append(loss_one_epoch / len(train_dataloader))
            train_acc.append(accuracy_one_epoch / len(train_dataloader))

            val_loss_, val_acc_ = validate_model(
                model, val_dataloader, criterion, device
            )
            val_loss.append(val_loss_)
            val_acc.append(val_acc_)

            model.train()
            scheduler.step(val_loss_)

            if val_loss_ < best_val_loss:
                best_val_loss = val_loss_

                if verbose:
                    print("Saving the new best model with val loss: ", best_val_loss)

                save_checkpoint(model, args, epoch, save_dir, device)

                # Save encoder to ONNX
                save_encoder_onnx(model, device, save_dir, args)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss[-1],
                    "train_accuracy": train_acc[-1],
                    "val_loss": val_loss[-1],
                    "val_accuracy": val_acc[-1],
                    "best_val_loss": best_val_loss,
                    "current_epoch": epoch,
                    "current_lr": scheduler._last_lr[0],
                },
                step=epoch,
            )

            # plot train and val loss, and train and val accuracy and save to MLflow as artifacts

            print(
                f"Train loss: {train_loss[-1] :.4f}, Train Accuracy: {train_acc[-1] :.4f}, "
                f"Val Loss: {val_loss[-1] :.4f}, Val Accuracy: {val_acc[-1] :.4f}, "
                f"Elapsed: {(time.time() - start_time) / 60 :.2f} min"
            )

    return train_loss, train_acc, val_loss, val_acc, best_val_loss


def validate_model(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(
            val_dataloader, leave=False, total=len(val_dataloader), desc="Validation"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)
            val_acc += (preds == labels).float().mean().item()

    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)

    return val_loss, val_acc


def save_checkpoint(
    model: torch.nn.Module,
    args: Namespace,
    epoch: int,
    save_dir: str,
    device: torch.device,
) -> None:
    """
    Save a model checkpoint in multiple formats (PyTorch, TorchScript, ONNX).

    Args:
        model (torch.nn.Module): The model to be saved.
        args (Namespace): Additional arguments used for naming the saved models.
        epoch (int): The current epoch number.
        save_dir (str): The directory where the models will be saved.
        device (torch.device): The device on which the example input tensor will be allocated.
    """

    # Save as PyTorch checkpoint
    model_name = f"{args.arch}_best.pth"
    model_save_path = osp.join(save_dir, model_name)

    model_params = {
        "model_state_dict": model.state_dict(),
        "epochs_trained": epoch,
    }

    save_model = {**model_params, **vars(args)}

    torch.save(save_model, model_save_path)

    # Save as TorchScript model
    model_script_path = osp.join(save_dir, f"{args.arch}_best.pt")
    example_input = torch.randn(1, 3, 224, 224).to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        traced_model = torch.jit.trace(model, example_input)
    traced_model.save(model_script_path)

    # Save as ONNX model
    model_onnx_path = osp.join(save_dir, f"{args.arch}_best.onnx")
    torch.onnx.export(
        model,
        example_input,
        model_onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def save_encoder_onnx(
    model: torch.nn.Module, device: torch.device, save_dir: str, args: Namespace
) -> None:
    """
    Save the encoder part of a model as an ONNX file.

    Args:
        model (torch.nn.Module): The model with an 'encoder' attribute.
        save_dir (str): The directory where the ONNX model will be saved.
        args (Namespace): Additional arguments used to determine input size.
    """

    save_path = osp.join(save_dir, f"best_encoder_{args.arch}.onnx")

    input_size = (1, 3, args.image_size[0], args.image_size[1])

    encoder = model.encoder
    example_input = torch.randn(*input_size).to(device)
    torch.onnx.export(
        encoder,
        example_input,
        save_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
