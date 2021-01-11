import torch
from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet


def get_model(model_name, device, num_classes):
    if model_name == 'EfficientNet':
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif model_name == 'ResNet34':
        model = models.resnet34(pretrained=False)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        # torch.nn.init.eye_(model.fc.weight)
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    return model.to(device)
