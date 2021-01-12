import torch
from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet
import os
import torchvision


class MyResNet(nn.Module):
    def __init__(self, model, pth=None, train=False, num_classes=None):
        super(MyResNet, self).__init__()
        if pth:
            model.load_state_dict(
                torch.load(os.path.join(os.path.dirname(__file__), pth)))

        self.layers = nn.Sequential(*list(model.children())[:-1])
        self.train = train
        if train:
            self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layers(x)
        if self.train:
            x = self.fc(x)
        else:
            x = x.flatten()
        return x


class MyEfficientNet(nn.Module):
    def __init__(self):
        super(MyEfficientNet, self).__init__()
        efficientnet = EfficientNet.from_pretrained('efficientnet-b6', num_classes=500)
        efficientnet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "checkpoints/best_efficientnet_model.pth")))


    def forward(self, x):
        x = self.layers(x)
        x = x.flatten()
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        model = torchvision.models.resnet18(pretrained=False)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "checkpoints/resnet18-5c106cde.pth")))

        model.eval()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten()


def get_model(model_name, device, train=False, num_classes=None):
    if model_name == 'EfficientNet':
        efficientnet = EfficientNet.from_pretrained('efficientnet-b6', num_classes=500)
        model = torch.nn.DataParallel(efficientnet)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "checkpoints/best_efficientnet_model.pth")))
        model.module._fc = nn.Linear(model.module._fc.in_features, 2304)
        # model = MyEfficientNet(efficientnet,pth="checkpoints/best_efficientnet_model.pth")
    elif model_name == 'ResNet18':
        model = ResNet18()
    elif model_name == 'ResNet34':
        resnet = models.resnet34(pretrained=True)
        model = MyResNet(resnet, train=train, num_classes=num_classes)
    elif model_name == 'ResNet50':
        resnet = models.resnet50(pretrained=True)
        model = MyResNet(resnet, train=train, num_classes=num_classes)
    elif model_name == 'ResNet101':
        resnet = models.resnet101(pretrained=True)
        model = MyResNet(resnet, train=train, num_classes=num_classes)
    else:
        resnet = models.resnet50(pretrained=True)
        model = MyResNet(resnet, train=train, num_classes=num_classes)
    # for param in model.parameters():
    #     param.requires_grad = False
    return model.to(device)
