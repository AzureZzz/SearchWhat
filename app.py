import torch
import numpy as np
import cv2
import os
import pandas as pd
import random

from PIL import Image
from tqdm import tqdm
from config import *
from flask import Flask, render_template,request
from models import get_model
from utils import get_img_tensor, show_net_structure, cosin_features,toTensor_operator,normalize_operator
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__, template_folder='templates', static_folder="static")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_name, device, vector_len)
vectors = np.load(f'{vector_path}/{dataset}/vectors.npy')
df = pd.read_csv(f'{vector_path}/{dataset}/names.csv')
names = df['names'].to_list()
image_names = os.listdir('static/dataset/oxbuild/')
n = len(image_names)


def compute():
    img = get_img_tensor('dataset/oxbuild/all_souls_000002.jpg', device)
    vector = model(img).cpu().detach().numpy()
    # sims = cosine_similarity(vector, vectors)
    # print(sims)
    # dists = []
    sims = []
    for v in vectors:
        sims.append(cosin_features(vector[0], v))
        # dists.append(np.linalg.norm(vector[0] - v))
    # print(dists)
    # print(sims)
    res = [(name, sim) for (name, sim) in zip(names, sims)]
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)


@app.route('/search', methods=['GET', 'POST'])
def search():
    img_file = request.files.get('upload_img')
    nparr = np.frombuffer(img_file.stream.read(), dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, image_size)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = normalize_operator(toTensor_operator(img)).float().cuda()
    img = img.unsqueeze(0)
    img.to(device)
    vector = model(img).cpu().detach().numpy()
    # sims = []
    dists = []
    for v in vectors:
        # sims.append(cosin_features(vector[0], v))
        dists.append(np.linalg.norm(vector[0] - v))
    # res = [(name, sim) for (name, sim) in zip(names, sims)]
    # res.sort(key=lambda x: x[1], reverse=True)
    res = [(name, dist) for (name, dist) in zip(names, dists)]
    res.sort(key=lambda x: x[1])
    best_res = f'static/dataset/oxbuild/{res[0][0]}'
    top4_res = []
    other_res = []
    for i in range(1,5):
        top4_res.append(res[i][0])
    for i in range(5,17):
        other_res.append(res[i][0])
    top4_res = [f'static/dataset/oxbuild/{x}' for x in top4_res]
    other_res = [f'static/dataset/oxbuild/{x}' for x in other_res]
    return render_template('index.html', best_res=best_res, top4_res=top4_res, other_res=other_res)


@app.route('/')
def index():
    best_res = f'static/dataset/oxbuild/{image_names[random.randint(0,n)]}'
    top4_res = []
    other_res = []
    for i in range(1, 5):
        top4_res.append(image_names[random.randint(0,n)])
    for i in range(5, 17):
        other_res.append(image_names[random.randint(0,n)])
    top4_res = [f'static/dataset/oxbuild/{x}' for x in top4_res]
    other_res = [f'static/dataset/oxbuild/{x}' for x in other_res]
    return render_template('index.html', best_res=best_res, top4_res=top4_res, other_res=other_res)


@app.route('/index')
def index1():
    best_res = f'static/dataset/oxbuild/{image_names[random.randint(0, n)]}'
    top4_res = []
    other_res = []
    for i in range(1, 5):
        top4_res.append(image_names[random.randint(0, n)])
    for i in range(5, 17):
        other_res.append(image_names[random.randint(0, n)])
    top4_res = [f'static/dataset/oxbuild/{x}' for x in top4_res]
    other_res = [f'static/dataset/oxbuild/{x}' for x in other_res]
    return render_template('index.html', best_res=best_res, top4_res=top4_res, other_res=other_res)


if __name__ == '__main__':
    app.run()
