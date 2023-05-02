import torch
from torch import nn
from typing import Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import os.path as osp
import copy


class Trainer:
    """
    A Trainer class to handle the training and validation of a PyTorch model.

    This class is designed to handle various training and validation tasks
    for a PyTorch model, including performing training and validation steps,
    saving model checkpoints, and visualizing losses. It is designed to be
    flexible and easily customizable to accommodate different training
    scenarios, model architectures, and optimization algorithms.

    Attributes
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    criterion : nn.Module
        The loss function used to calculate the difference between predictions
        and target labels during training and validation.
    optimizer : torch.optim.Optimizer
        The optimization algorithm used to update the model's parameters.
    num_epochs : int
        The number of times the entire dataset is passed through the model
        during training.
    output_dir : str
        The directory where training outputs, such as model checkpoints and
        loss plots, are saved.
    architecture : str
        The name of the model architecture.
    config : dict, optional
        A configuration dictionary containing additional settings for the
        training process, such as architecture name, learning rate, etc.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        A learning rate scheduler that adjusts the learning rate during
        training, typically based on the number of epochs.
    train_dataloader : torch.utils.data.DataLoader, optional
        A DataLoader that provides batches of training data.
    val_dataloader : torch.utils.data.DataLoader, optional
        A DataLoader that provides batches of validation data.
    patience : int, optional
        The number of epochs to wait for improvement in validation loss before
        stopping training early. If None, early stopping is not used.
    device : str
        The device on which to perform training and validation, such as 'cpu'
        or 'cuda'.

    Methods
    -------
    training_step(inputs, labels)
        Performs a single training step, including forward pass, loss calculation,
        backward pass, and optimizer step.
    train_epoch(train_dataloader)
        Trains the model for one complete pass through the dataset using the
        provided DataLoader.
    validation_step(inputs, labels)
        Performs a single validation step, including forward pass and loss calculation.
    val_epoch(val_dataloader)
        Validates the model for one complete pass through the dataset using the
        provided DataLoader.
    save_checkpoint(epoch, save_dir)
        Saves the current model checkpoint, including model state, optimizer state,
        and other training settings.
    train_epochs(num_epochs, train_dataloader, val_dataloader)
        Trains the model for a specified number of epochs, performing validation
        at the end of each epoch if a validation DataLoader is provided.
    plot_losses()
        Generates and saves a plot of training and validation losses over time.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        output_dir: str = "training_output",
        architecture: str = "default",
        # Optional parameters
        config: dict = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        patience: int = 10,
        device: str = "cpu",
        save_best_model: bool = True,
    ):

        self.model = model
        self.architecture = architecture
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.image_size = (224, 224)

        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.config = config

        self.patience = patience

        self.output_dir = output_dir

        # Set model to device
        self.model.to(self.device)

        if self.config is not None:
            self.num_epochs = self.config["num_epochs"]
            self.patience = self.config["patience"]
            self.output_dir = self.config["output_dir"]
            self.architecture = self.config["architecture"]

        # Variables that will change during training
        self.train_losses = []
        self.val_losses = []
        self.current_patience = 0

        # model outputs for training and validation
        self.train_outputs = []
        self.train_gt = []
        self.val_outputs = []
        self.val_gt = []

        #
        self.best_val_loss = np.inf
        self.best_train_loss = np.inf

        self.save_best_model = save_best_model

    def training_step(self, inputs, labels):
        """
        Perform a single training step, including forward pass, loss calculation,
        backward pass, and optimizer step.

        Parameters
        ----------
        inputs : torch.Tensor
            A batch of input data.
        labels : torch.Tensor
            A batch of target labels corresponding to the input data.

        Notes
        -----
        This method performs the following operations in order:
        1. Set the model to training mode.
        2. Perform a forward pass through the model using the input data.
        3. Calculate the loss using the model's predictions and the target labels.
        4. Perform a backward pass to compute gradients of the loss with respect to model parameters.
        5. Update the model's parameters using the optimizer.
        6. Zero the gradients to prevent accumulation across iterations.
        """

        # 1. Model in training mode
        self.model.train()

        # 2. Forward pass
        logits = self.model(inputs)

        # 3. Loss
        loss = self.criterion(logits, labels)

        # 4. Backward Pass
        loss.backward()

        # 5. Optimizer step
        self.optimizer.step()

        # 6. Zero grad
        self.optimizer.zero_grad()

        # save model outputs
        train_outputs = logits.detach().cpu().numpy()
        train_outputs = np.argmax(train_outputs, axis=1)
        self.train_outputs.extend(train_outputs.flatten())
        self.train_targets.extend(labels.detach().cpu().numpy().flatten())

        return loss.item()

    def train_epoch(
        self, train_dataloader: Optional[torch.utils.data.DataLoader] = None
    ):
        """
        Train the model for one complete pass through the dataset using the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader, optional
            A DataLoader that provides batches of training data. If not provided,
            the method will use the instance's train_dataloader attribute.

        Returns
        -------
        float
            The average training loss for the epoch.

        Raises
        ------
        ValueError
            If no train_dataloader is provided and the instance's train_dataloader attribute is None.

        Notes
        -----
        This method performs the following operations:
        1. Set the model to training mode.
        2. Initialize an epoch loss variable to accumulate losses during the epoch.
        3. Iterate through the DataLoader, fetching batches of input data and labels.
        4. Move input data and labels to the appropriate device.
        5. Call the `training_step` method to perform a single training step.
        6. Calculate the loss for the current batch and update the epoch loss.
        7. After iterating through the DataLoader, compute the average epoch loss.
        """
        ...
        self.model.train()
        epoch_loss = 0

        # reset model outputs
        self.train_outputs = []
        self.train_targets = []

        if self.train_dataloader is not None:
            train_dataloader = self.train_dataloader
        elif self.train_dataloader is None and train_dataloader is not None:
            train_dataloader = train_dataloader
        else:
            raise ValueError("Train dataloader is not provided")

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss = self.training_step(inputs, labels)

            epoch_loss += loss

        epoch_loss /= len(train_dataloader)

        self.train_losses.append(epoch_loss)

        # update best train loss
        if epoch_loss < self.best_train_loss:
            self.best_train_loss = epoch_loss

        return epoch_loss

    def validation_step(self, inputs, labels):

        """
        Perform a single validation step, including forward pass and loss calculation.

        Parameters
        ----------
        inputs : torch.Tensor
            A batch of input data.
        labels : torch.Tensor
            A batch of target labels corresponding to the input data.

        Returns
        -------
        float
            The loss value for the current batch.

        Notes
        -----
        This method performs the following operations:
        1. Disable gradient calculation to save memory and computation during validation.
        2. Perform a forward pass through the model using the input data.
        3. Calculate the loss using the model's predictions and the target labels.
        """

        with torch.no_grad():
            # 1. Forward pass
            logits = self.model(inputs)

            # 2. Loss
            loss = self.criterion(logits, labels)

            # 3. Save outputs by detaching and converting to numpy and shifting to cpu
            val_outputs = logits.detach().cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1)

            self.val_outputs.extend(val_outputs.flatten())
            self.val_targets.extend(labels.detach().cpu().numpy().flatten())

        return loss.item()

    def val_epoch(self, val_dataloader: Optional[torch.utils.data.DataLoader] = None):
        """
        Validate the model for one complete pass through the dataset using the provided DataLoader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader, optional
            A DataLoader that provides batches of validation data. If not provided,
            the method will use the instance's val_dataloader attribute.

        Returns
        -------
        float
            The average validation loss for the epoch.

        Raises
        ------
        ValueError
            If no val_dataloader is provided and the instance's val_dataloader attribute is None.

        Notes
        -----
        This method performs the following operations:
        1. Set the model to evaluation mode.
        2. Initialize an epoch loss variable to accumulate losses during the epoch.
        3. Iterate through the DataLoader, fetching batches of input data and labels.
        4. Move input data and labels to the appropriate device.
        5. Call the `validation_step` method to perform a single validation step.
        6. Update the epoch loss with the loss for the current batch.
        7. After iterating through the DataLoader, compute the average epoch loss.
        """

        self.model.eval()
        epoch_loss = 0

        # reset val outputs
        self.val_outputs = []
        self.val_targets = []

        if self.val_dataloader is not None:
            val_dataloader = self.val_dataloader
        elif self.val_dataloader is None and val_dataloader is not None:
            val_dataloader = val_dataloader
        else:
            raise ValueError("Val dataloader is not provided")

        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss = self.validation_step(inputs, labels)
            epoch_loss += loss

        epoch_loss /= len(val_dataloader)

        self.val_losses.append(epoch_loss)

        # update best model
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            if self.save_best_model:
                self.save_checkpoint(save_dir=self.output_dir)
                self.save_onnx_model(save_dir=self.output_dir)
                self.save_torchscript_model(save_dir=self.output_dir)

        elif epoch_loss > self.best_val_loss:
            self.current_patience += 1

        return epoch_loss

    def save_checkpoint(self, save_dir: Optional[str] = None):

        """
        Save the current model checkpoint, including model parameters and training state.

        Parameters
        ----------
        save_dir : str, optional
            The directory to save the checkpoint. If not provided, the method will use the
            instance's output_dir attribute.

        Notes
        -----
        This method performs the following operations:
        1. Create a dictionary of model parameters and training state information.
        2. Save the model parameters and training state as a PyTorch checkpoint (.pth) file.
        """

        if save_dir is None:
            save_dir = self.output_dir

        # make output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        model_name = f"{self.architecture}.pth"
        model_save_path = osp.join(save_dir, model_name)

        model_state_dict = self.model.state_dict()

        model_params = {
            "model_state_dict": model_state_dict,
            "architecture": self.architecture,
        }
        save_model = model_params.copy()

        if self.config is not None:
            save_model.update(self.config)

        # save as pytorch checkpoint
        torch.save(save_model, model_save_path)

        # save as torchscript model
        model_save_path = osp.join(save_dir, f"{self.architecture}.pt")
        torch.jit.save(torch.jit.script(self.model), model_save_path)

        return None

    def save_torchscript_model(self, save_dir: Optional[str] = None):

        """
        Save the torchscript model checkpoint, including model parameters and training state.

        Parameters
        ----------
        save_dir : str, optional
            The directory to save the checkpoint. If not provided, the method will use the
            instance's output_dir attribute.

        Notes
        -----
        This method performs the following operations:
        1. Create a dictionary of model parameters and training state information.
        2. Save the model as a TorchScript (.pt) file for deployment purposes.
        """

        if save_dir is None:
            save_dir = self.output_dir

        # save as torchscript model
        model_save_path = osp.join(save_dir, f"{self.architecture}.pt")
        torch.jit.save(torch.jit.script(self.model), model_save_path)

        return None

    def save_onnx_model(
        self,
        save_dir: Optional[str] = None,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ):

        """
        Save the current model checkpoint as an ONNX model, including model parameters and training state.

        Parameters
        ----------
        epoch : int
            The number of epochs completed when saving the checkpoint.
        save_dir : str, optional
            The directory to save the checkpoint. If not provided, the method will use the
            instance's output_dir attribute.

        Notes
        -----
        This method performs the following operations:
        1. Create a dictionary of model parameters and training state information.
        2. Save the model as a ONNX (.onnx) file for deployment purposes.
        """

        if save_dir is None:
            save_dir = self.output_dir

        model_save_path = osp.join(save_dir, f"{self.architecture}.onnx")

        # create a dummy input for tracing
        image = torch.randn(input_shape).to(self.device)

        # save as ONNX model
        torch.onnx.export(
            self.model,  # model being run
            image,  # model input (or a tuple for multiple inputs)
            model_save_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )

        return None

    def train_epochs(
        self,
        num_epochs: Optional[int] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):

        """
        Train the model for multiple epochs, and optionally validate the model.

        Parameters
        ----------
        num_epochs : int, optional
            The number of epochs to train the model. If not provided, the method will use
            the instance's num_epochs attribute.
        train_dataloader : torch.utils.data.DataLoader, optional
            A DataLoader that provides batches of training data. If not provided,
            the method will use the instance's train_dataloader attribute.
        val_dataloader : torch.utils.data.DataLoader, optional
            A DataLoader that provides batches of validation data. If not provided,
            the method will use the instance's val_dataloader attribute.

        Notes
        -----
        This method performs the following operations:
        1. Iterate through the specified number of epochs.
        2. Train the model for one epoch using the train_epoch method.
        3. If a validation DataLoader is provided, validate the model using the val_epoch method.
        4. Update the training and validation losses.
        5. If early stopping is enabled, stop training if there is no improvement in validation loss.
        6. Save the model checkpoint if the validation loss has improved.
        """

        if num_epochs is not None:
            self.num_epochs = num_epochs

        with tqdm(
            range(self.num_epochs), desc="Epochs", unit="epoch"
        ) as epoch_progress:
            for epoch in epoch_progress:
                train_loss = self.train_epoch(train_dataloader)

                if self.val_dataloader or val_dataloader:
                    val_loss = self.val_epoch(val_dataloader)
                    epoch_progress.set_postfix(
                        {
                            "Train Loss": f"{train_loss:.4f}",
                            "Val Loss": f"{val_loss:.4f}",
                        }
                    )
                else:
                    epoch_progress.set_postfix({"Train Loss": f"{train_loss:.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        if self.current_patience >= self.patience:
            print(
                f"Early stopping at epoch {epoch} due to no improvement in validation loss."
            )
            return

    def plot_losses(self):
        """
        Plot the training and validation losses and save the figure as an image file.

        Notes
        -----
        This method performs the following operations:
        1. Check if there are recorded training losses. If not, print an error message and return.
        2. Create a new figure to plot the training and validation losses.
        3. Plot the training losses over the number of epochs.
        4. If there are recorded validation losses, plot them over the number of epochs.
        5. Label the axes and add a legend.
        6. Save the figure as an image file (losses.png) in the output directory.
        """

        if not self.train_losses:
            print("No training losses recorded.")
            return

        plt.figure(figsize=(5, 5))
        plt.plot(self.train_losses, label="Train Loss")

        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Losses vs. Epochs")
        plt.legend()

        # create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        plt.savefig(os.path.join(self.output_dir, "losses.png"), dpi=300)

        plt.close("all")
