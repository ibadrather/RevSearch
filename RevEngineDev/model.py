import os
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
import timm


class RevSearchFeatureExtractorVGG16(nn.Module):
    """
    Reverse Search Feature Extractor based on VGG16 architecture.

    This model is designed to extract features from images using a modified
    VGG16 architecture. The features can be used for various tasks, such as
    image classification, object detection, or reverse image search.

    Attributes:
        num_classes (int): Number of output classes for the final layer.
        dropout (float): Dropout rate used in the model.
        feature_vector_size (int): Size of the feature vector produced by the encoder.
        features (nn.Sequential): Feature extraction layers from the VGG16 model.
        pooling (nn.Module): Average pooling layer from the VGG16 model.
        encoder (nn.Sequential): Encoder part of the model, responsible for feature extraction.
        decoder (nn.Sequential): Decoder part of the model, responsible for classification.
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.2,
        feature_vector_size: int = 300,
    ):
        super(RevSearchFeatureExtractorVGG16, self).__init__()

        # Load the pretrained VGG16 model
        pretrained_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        features = list(pretrained_model.features)

        self.num_classes = num_classes

        # Tunable parameters
        self.dropout = dropout
        self.feature_vector_size = feature_vector_size

        # Freeze training for all layers but the last 4
        for feature in features[:-4]:
            for param in feature.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*features)

        self.pooling = pretrained_model.avgpool

        self.encoder = nn.Sequential(
            self.features,
            self.pooling,
            nn.Flatten(),
            nn.Linear(25088, 3000),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(3000, 1500),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1000, self.feature_vector_size),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_vector_size, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class RevSearchFeatureExtractorResNet(nn.Module):
    def __init__(self, num_classes, feature_vector_size=500, dropout=0.2):
        super(RevSearchFeatureExtractorResNet, self).__init__()

        self.feature_vector_size = feature_vector_size
        self.dropout = dropout
        self.num_classes = num_classes

        # Load the pretrained ResNet50 model
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Access the ResNet components and freeze training for conv1
        for param in pretrained_model.conv1.parameters():
            param.requires_grad = False

        # for param in pretrained_model.layer1.parameters():
        #     param.requires_grad = False

        for param in pretrained_model.layer2.parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(
            pretrained_model.conv1,
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            pretrained_model.layer1,
            pretrained_model.layer2,
            pretrained_model.layer3,
            pretrained_model.layer4,
            pretrained_model.avgpool,
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.feature_vector_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.feature_vector_size, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


import timm
import torch.nn as nn


class RevSearchFeatureExtractorEfficientNet(nn.Module):
    def __init__(self, num_classes, feature_vector_size=500, dropout=0.2):
        super(RevSearchFeatureExtractorEfficientNet, self).__init__()

        self.feature_vector_size = feature_vector_size
        self.dropout = dropout
        self.num_classes = num_classes

        # Load the pretrained EfficientNet model
        pretrained_model = timm.create_model("efficientnet_b0", pretrained=True)

        # Access the EfficientNet components and freeze training for the first layer
        for param in pretrained_model.conv_stem.parameters():
            param.requires_grad = False

        for param in pretrained_model.blocks[0].parameters():
            param.requires_grad = False

        # Remove the classifier from the pretrained model
        pretrained_model.classifier = nn.Identity()

        # Add custom layers to the encoder
        self.encoder = nn.Sequential(
            pretrained_model,
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.feature_vector_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.feature_vector_size, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def main() -> None:
    os.system("clear")
    from torchinfo import summary

    model = RevSearchFeatureExtractorEfficientNet(
        num_classes=196, dropout=0.2, feature_vector_size=300
    )

    # model encoder
    model_encoder = model.encoder

    summary(model_encoder, input_size=(1, 3, 224, 224), device="cpu")


if __name__ == "__main__":
    main()
