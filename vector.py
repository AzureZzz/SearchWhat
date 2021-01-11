import torch
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from models import get_model
from config import *
from utils import get_img_tensor


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, device, vector_len)
    image_list = os.listdir(f'dataset/{dataset}')
    vectors = []
    for image in tqdm(image_list):
        img = get_img_tensor(f'dataset/{dataset}/{image}', device)
        v = model(img)
        vectors.append(v.cpu().detach().numpy()[0])
    save_path = f'{vector_path}/{dataset}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(f'{save_path}/vectors', np.array(vectors))
    df = pd.DataFrame({'names':image_list})
    df.to_csv(f'{save_path}/names.csv')
