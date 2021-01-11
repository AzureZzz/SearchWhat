import torch
from torchvision import models
from torch import nn


def get_model(model_name, device, num_classes):
    if model_name == 'ResNet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        torch.nn.init.eye_(model.fc.weight)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        torch.nn.init.eye_(model.fc.weight)
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        torch.nn.init.eye_(model.fc.weight)
    else:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        torch.nn.init.eye_(model.fc.weight)
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)
