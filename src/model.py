import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2, pretrained=True):

    # Charge ResNet-18 pré-entraîné
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Gèle toutes les couches
    for param in model.parameters():
        param.requires_grad = False

    # Remplace la dernière couche
    model.fc = nn.Linear(512, num_classes)

    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")
    return device