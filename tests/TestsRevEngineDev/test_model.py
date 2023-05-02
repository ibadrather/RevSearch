import pytest
import torch
from torchvision.models import vgg16, VGG16_Weights

import os
import sys

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from RevEngineDev.model import RevSearchFeatureExtractor


@pytest.fixture
def mock_vgg16(mocker):
    mock_vgg16 = mocker.MagicMock()
    mocker.patch("torchvision.models.vgg16", return_value=mock_vgg16)
    return mock_vgg16


def test_rev_search_feature_extractor_forward(mock_vgg16):
    num_classes = 5
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    model = RevSearchFeatureExtractor(num_classes=num_classes)
    output = model(input_tensor)

    assert output.size() == (batch_size, num_classes)


def test_rev_search_feature_extractor_frozen_layers(mock_vgg16):
    num_classes = 5
    model = RevSearchFeatureExtractor(num_classes=num_classes)

    # Check that the layers are frozen except for the last 4
    for i, feature in enumerate(model.features[:-4]):
        for param in feature.parameters():
            assert not param.requires_grad, f"Layer {i} should be frozen"
    for i, feature in enumerate(model.features[-4:], start=len(model.features) - 4):
        for param in feature.parameters():
            assert param.requires_grad, f"Layer {i} should not be frozen"


def test_rev_search_feature_extractor_feature_vector_size(mock_vgg16):
    num_classes = 5
    feature_vector_size = 300
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    model = RevSearchFeatureExtractor(
        num_classes=num_classes, feature_vector_size=feature_vector_size
    )
    feature_vector = model.encoder(input_tensor)

    assert feature_vector.size() == (batch_size, feature_vector_size)
