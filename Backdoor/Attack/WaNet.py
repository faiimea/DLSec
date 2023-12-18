from . import base
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

transform_PIL2Tensor = transforms.ToTensor()
transform_Tensor2PIL = transforms.ToPILImage()


def gen_grid(height, k):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.upsample(ins, size=height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return identity_grid, noise_grid


def adjust_tensor(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
        return tensor
    elif len(tensor.shape) == 3:
        if tensor.shape[2] < 5:
            tensor = tensor.permute(2, 0, 1)
            return tensor
        else:
            return tensor
    else:
        print("You're passing a 4 dimension tensor to function [adjust_tensor], which was supposed to process a single tensor")
        return tensor


def create_poisoned_data(dataset):
    class Poisoned_data(dataset):
        def __init__(self, root, transform, poison_rate, isTrain, target_label, WAgrid, noise_ratio):
            super().__init__(root, train=isTrain, transform=transform, download=True)
            self.width, self.height, self.channels = self.__shape_info__()
            self.target_label = target_label
            self.WAgrid = WAgrid
            self.poison_rate = poison_rate if isTrain else 1.0
            self.noise_rate = noise_ratio if isTrain else 0
            assert self.poison_rate + self.noise_rate <= 1, "中毒数据与加噪扭曲数据总占比超过1，请调整参数设置"

            indices = range(len(self.targets))
            self.poi_indices = random.sample(indices, k=int(len(indices) * self.poison_rate))
            self.noise_indices = random.sample(list(set(indices) - set(self.poi_indices)), k=int(len(indices) * self.noise_rate))

        def __shape_info__(self):
            if len(self.data.shape[1:]) == 3:
                return self.data.shape[1:]
            elif len(self.data.shape[1:]) == 2:
                return self.data.shape[1:][0], self.data.shape[1:][1], 2

        def __getitem__(self, index):
            # img, target = self.data[index], self.targets[index]
            img, target = self.data[index], self.targets[index]
            if self.transform is not None:
                img = self.transform(img)

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            elif isinstance(img, Image.Image):
                img = transform_PIL2Tensor(img)
            elif isinstance(img, torch.Tensor):
                pass
            else:
                raise TypeError("数据类型不支持，请检查")
            img = adjust_tensor(img)

            if index in self.poi_indices:
                target = self.target_label
                img = F.grid_sample(img.unsqueeze(0), self.WAgrid, align_corners=True).squeeze()
            elif index in self.noise_indices:
                ins = torch.rand(1, self.height, self.height, 2) * 2 - 1
                noise_grid = torch.clamp(self.WAgrid + ins / self.height, -1, 1)
                img = F.grid_sample(img.unsqueeze(0), noise_grid, align_corners=True).squeeze()
            else:
                img = img.squeeze()

            if self.channels == 2:
                img = img.unsqueeze(0)

            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            return img, target

    return Poisoned_data


class WaNet(base.BackdoorAttack):
    """
        WaNet Args:
            s (float):扭曲网格与原网格比值，建议小于0.75
            k (int):扭曲网格生成时所用参数，建议小于6
            noise_ratio(float):为隐藏攻击模式而使用的噪声扭曲网格在总数据中的占比
        """

    def __init__(self, tag: str = 'CustomModel', device: str = 'cpu', model=None, dataset=None,data_download_path=None, poison_rate: float = 0.05, lr: float = 0.1, target_label=2, epochs: int = 20, batch_size: int = 64, optimizer: str = 'sgd', criterion=None, local_model_path: str = None,WAgrid_path:str=None, s: float = 0.5, k: int = 4,
                 noise_ratio: float = 0.2):
        super().__init__(tag, device, model, dataset,data_download_path, poison_rate, lr, target_label, epochs, batch_size, optimizer, criterion, local_model_path)
        if WAgrid_path is not None:
            if os.path.exists(WAgrid_path):
                self.WAgrid=torch.load(WAgrid_path)
            else:
                self.identity_grid, self.elastic_grid = gen_grid(int(self.data_shape[0]), k)
                grid = self.identity_grid + s * self.elastic_grid / self.data_shape[0]
                self.WAgrid = torch.clamp(grid, -1, 1)
                torch.save(self.WAgrid, WAgrid_path)
        else:
            self.identity_grid, self.elastic_grid = gen_grid(int(self.data_shape[0]), k)
            grid = self.identity_grid + s * self.elastic_grid / self.data_shape[0]
            self.WAgrid = torch.clamp(grid, -1, 1)

        poisoned_train_data = create_poisoned_data(dataset)(self.data_path, self.transform, poison_rate, True, target_label, self.WAgrid, noise_ratio)
        poisoned_test_data = create_poisoned_data(dataset)(self.data_path, self.transform, poison_rate, False, target_label, self.WAgrid, 0)
        self.dataloader_train = DataLoader(poisoned_train_data, batch_size=self.batch_size, shuffle=True)
        self.dataloader_cleantest = DataLoader(self.clean_testdata, batch_size=self.batch_size, shuffle=True)
        self.dataloader_poisonedtest = DataLoader(poisoned_test_data, batch_size=self.batch_size, shuffle=True)

    def train(self,epochs=None):
        if epochs is None:
            train_epochs=self.epochs
        else:
            train_epochs=epochs
        print(f"Training on {self.device} for {train_epochs} epochs\n")
        min_loss = np.inf
        for epoch in range(train_epochs):
            train_stats = self.train_one_epoch(self.dataloader_train, self.model, self.optimizer, self.criterion, self.device)
            test_stats = self.evaluate_model(self.dataloader_cleantest, self.dataloader_poisonedtest, self.model, self.criterion, self.device)
            print(f"EPOCH {epoch + 1}/{self.epochs}   loss: {train_stats['loss']:.4f} CDA: {test_stats['CDA']:.4f}, ASR: {test_stats['ASR']:.4f}\n")
            if train_stats['loss'] < min_loss:
                print("Model updated ---", test_stats)
                torch.save(self.model, self.model_path)
                min_loss = train_stats['loss']

    def test(self):
        test_stats = self.evaluate_model(self.dataloader_cleantest, self.dataloader_poisonedtest, self.model, self.criterion, self.device)
        print(f"CDA: {test_stats['CDA']:.4f}, ASR: {test_stats['ASR']:.4f}\n")

    def display(self):
        for batch in self.dataloader_poisonedtest:
            one_img = np.transpose(self.detransform(batch[0][0]).numpy(), (1, 2, 0))
            plt.imshow(one_img)
            plt.axis('off')
            plt.show()
            return
