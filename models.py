import torch
from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet
import os
import torchvision


class MyResNet(nn.Module):
    def __init__(self, model):
        super(MyResNet,self).__init__()
        self.res_layers = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.res_layers(x)
        return x.flatten()

class query_net(nn.Module):
   def __init__(self):
      super(query_net, self).__init__()

      # model = torchvision.models.vgg16(pretrained=True)
      # model = torchvision.models.resnet18(pretrained=False)
      model = torchvision.models.resnet18(pretrained=False)  # 读取网络
      model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "checkpoints/resnet18-5c106cde.pth")))
      # 读取网络参数
      #resnet34-333f7ec4.pth
      #resnet18-5c106cde.pth

      model.eval()  # 测试状态，BN层与训练状态有区别

      #self.features = model.features
      #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

      # resnet18
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
      # x = self.features(x)
      # x = self.avgpool(x)

      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)

      #x = x.view(-1)
      x = torch.flatten(x)  # 尽量用flatten
      return x


def get_model(model_name, device):
    if model_name == 'EfficientNet':
        efficientnet = EfficientNet.from_pretrained('efficientnet-b6')
        model = MyResNet(efficientnet)
    elif model_name == 'ResNet18':
        # resnet = models.resnet18(pretrained=False)
        # model = MyResNet(resnet)
        model = query_net()
        # model.load_state_dict(torch.load('checkpoints/resnet18-5c106cde.pth'))
    elif model_name == 'ResNet34':
        resnet = models.resnet34(pretrained=True)
        model = MyResNet(resnet)
    elif model_name == 'ResNet50':
        resnet = models.resnet50(pretrained=True)
        model = MyResNet(resnet)
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)
