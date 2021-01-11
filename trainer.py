import os
import torch
import argparse
import logging
import numpy as np
import random

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, char_color
from torchvision.transforms import *
# from albumentations import (
#     HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomResizedCrop,
#     RandomBrightnessContrast, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
# )
#
# from albumentations.pytorch import ToTensorV2

from loader import get_dataset
from models import get_model


class Trainer(object):

    def __init__(self, model, train_loader, val_loader, args, device, logging):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.logging = logging

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                    patience=3, verbose=True, min_lr=1e-5)
        if args.action == 'train':
            self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            self.inputs = next(iter(train_loader))[0]
            self.writer.add_graph(model, self.inputs.to(device, dtype=torch.float32))
        if args.DataParallel:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

    def train(self):
        epochs = self.args.epochs
        n_train = len(self.train_loader.dataset)
        step = 0
        best_acc = 0.
        accs = AverageMeter()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            # training
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, labels = batch[0], batch[1]

                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.long)
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)

                    epoch_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    accs.update((preds.argmax(1) == labels).sum().item() / images.size(0), images.size(0))
                    pbar.set_postfix(**{'loss': loss.item(), 'acc': accs.avg})
                    self.writer.add_scalar('acc/train', accs.avg, step)
                    self.writer.add_scalar('Loss/train', loss.item(), step)
                    pbar.update(images.shape[0])
                    step = step + 1
            # eval
            if (epoch + 1) % self.args.val_epoch == 0:
                acc = self.test(mode='val')
                if acc > best_acc:
                    best_acc = acc
                    if self.args.save_path:
                        if not os.path.exists(self.args.save_path):
                            os.makedirs(self.args.save_path)
                        torch.save(self.model.state_dict(), f'{self.args.save_path}/best_model.pth')
                        self.logging.info(char_color(f'best model saved !', word=33))

                self.logging.info(f'acc: {acc}')
                self.writer.add_scalars('Valid', {'acc': acc}, step)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
                self.scheduler.step(acc)
            if (epoch + 1) % self.args.save_model_epoch == 0:
                if self.args.save_path:
                    if not os.path.exists(self.args.save_path):
                        os.makedirs(self.args.save_path)
                    model_name = f'{self.args.task}_'
                    torch.save(self.model.state_dict(), f'{self.args.save_path}/{model_name}{epoch + 1}.pth')
                    self.logging.info(char_color(f'Checkpoint {epoch + 1} saved !'))
        self.writer.close()

    def test(self, mode='val', model_path=None, aug=False):
        self.model.train(False)
        self.model.eval()

        accs = AverageMeter()
        test_len = len(self.val_loader)
        step = 0
        with torch.no_grad():
            with tqdm(total=test_len, desc=f'{mode}', unit='batch') as pbar:
                for batch in self.val_loader:
                    images, labels = batch[0], batch[1]
                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.long)
                    preds = self.model(images)
                    accs.update((preds.argmax(1) == labels).sum().item() / images.size(0), images.size(0))
                    pbar.set_postfix(**{'acc': accs.avg})
                    pbar.update(images.shape[0])
                    step = step + 1
        return accs.avg


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_logging():
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s\n%(levelname)s:%(message)s'
        format='%(levelname)s:%(message)s'
    )
    return logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=10)

    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--action', type=str, default='train')

    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--val_epoch', type=int, default=2)
    parser.add_argument('--save_model_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--data_path', type=str, default='static/dataset')
    parser.add_argument('--tensorboard_dir', type=str, default='result/runs')
    parser.add_argument('--save_path', type=str, default='checkpoints')

    parser.add_argument('--cuda_ids', type=str, default='0')
    parser.add_argument('--DataParallel', type=bool, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    logging = get_logging()
    seed_everything(args.seed)

    transforms_train = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])
    transforms_val = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    train_loader, val_loader = get_dataset(args.data_path, args.batch_size, args.batch_size_val, transforms_train,
                                           transforms_val)
    net = get_model('EfficientNet', device, 500)

    trainer = Trainer(net, train_loader, val_loader, args, device, logging)
    trainer.train()


if __name__ == '__main__':
    main()