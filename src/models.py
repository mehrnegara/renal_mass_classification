import torch.nn as nn
import torchvision.models as models

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

class ResNetSimCLR(nn.Module):
    """
    Used only in IMAGE mode — identical to your original.
    """
    def __init__(self, base_model, dim_in, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None, num_classes=out_dim),
            "resnet50": models.resnet50(weights=None, num_classes=out_dim)
        }

        self.backbone = self._get_basemodel(base_model)
        first_conv = self.backbone.conv1

        # Replace first conv layer to accept custom number of channels (dim_in)
        self.backbone.conv1 = nn.Conv2d(
            dim_in,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Pass one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class MLPClassifier(nn.Module):
    """
    Used only in EMBEDDINGS mode — EXACTLY as you requested.
    Linear(512 → 3), nothing extra.
    """
    def __init__(self, dim_in=512, out_dim=3):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(dim_in, out_dim, bias=True)  # EXACT MATCH

    def forward(self, x):
        return self.fc(x)
