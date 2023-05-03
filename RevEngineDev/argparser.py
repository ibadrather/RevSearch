import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training a facial recognition model.

    Returns:
        argparse.Namespace: An object containing parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train a facial recognition model.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs.")
    parser.add_argument("--bs", type=int, default=50, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate.")
    parser.add_argument(
        "--num_classes", type=int, default=100, help="Number of classes."
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping."
    )
    parser.add_argument(
        "--image_size", nargs=2, type=int, default=(224, 224), help="Image size."
    )

    parser.add_argument(
        "--output_dir", type=str, default="training_output", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training."
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to use for training."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ReduceLROnPlateau",
        help="Scheduler to use for training.",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="CrossEntropyLoss",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--arch", type=str, default="efficientnet", help="Model architecture."
    )
    # feature vector size argument
    parser.add_argument(
        "--feature_vector_size",
        type=int,
        default=700,
        help="Size of the feature vector.",
    )
    # weight decay
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer."
    )
    # momentum
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for optimizer."
    )
    return parser.parse_known_args()[0]


"""
May 3, 2023, 1.17pm
    Best Trial:
        Value:  1.6014668927636257
        Params: {'bs': 30, 'feature_vector_size': 800, 'lr': 4.6e-05}
"""
