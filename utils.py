import cv2
import numpy as np
import torch
from config import *
from torchvision import transforms

toTensor_operator = transforms.ToTensor()
normalize_operator = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def show_net_structure(net):
    for name, module in net._modules.items():
        print(name, ":", module)


def get_img_tensor(path, device):
    img = cv2.imread(path)
    img = cv2.resize(img, image_size)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = normalize_operator(toTensor_operator(img)).float().cuda()
    img = img.unsqueeze(0)
    return img.to(device)


def cosin_features(feature1, feature2):
    feature1 = np.array(feature1, dtype=np.float32)
    feature2 = np.array(feature2, dtype=np.float32)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    similarity = np.dot(feature1, feature2.T) / (norm1 * norm2)
    return similarity
