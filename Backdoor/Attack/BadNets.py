import torch
from torch.utils.data import DataLoader
from . import base
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

def implement_trigger(mode,clean_img, original_pic_width, original_pic_height, trigger_pic_path, trigger_size):
    trigger_image = Image.open(trigger_pic_path).convert(mode).resize((trigger_size[0], trigger_size[1]))
    clean_img.paste(trigger_image, (original_pic_width - trigger_size[0], original_pic_height - trigger_size[1]))
    return clean_img


def create_poisoned_data(dataset):
    class Poisoned_data(dataset):
        def __init__(self, root, transform, poison_rate, isTrain, target_label, trigger_path, trigger_size):
            super().__init__(root, train=isTrain, transform=transform, download=True)
            self.width, self.height, self.channels = self.__shape_info__()
            self.target_label = target_label
            self.trigger_path = trigger_path
            self.trigger_size = trigger_size

            self.poison_rate = poison_rate if isTrain else 1.0
            indices = range(len(self.targets))
            self.poi_indices = random.sample(indices, k=int(len(indices) * self.poison_rate))

        def __shape_info__(self):
            if len(self.data.shape[1:]) == 3:
                return self.data.shape[1:]
            elif len(self.data.shape[1:]) == 2:
                return self.data.shape[1:][0], self.data.shape[1:][1], 2

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                img = Image.fromarray(img.numpy())
            elif isinstance(img, Image.Image):
                pass
            else:
                raise TypeError("数据类型不支持，请检查")

            if index in self.poi_indices:
                target = self.target_label
                if self.channels>2:
                    img = implement_trigger('RGB',img, self.width, self.width, self.trigger_path, self.trigger_size)
                else:
                    img = implement_trigger('L',img, self.width, self.width, self.trigger_path, self.trigger_size)

            if self.transform is not None:
                img = self.transform(img)

            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            return img, target

    return Poisoned_data


class Badnets(base.BackdoorAttack):
    def __init__(self, tag: str = 'CustomModel', device: str = 'cpu', model=None, dataset=None, poison_rate: float = 0.05, lr: float = 0.1, target_label=2, epochs: int = 20, batch_size: int = 64, optimizer: str = 'sgd', criterion=None, local_model_path: str = None,
                 trigger_path: str = None, trigger_size: tuple = (5, 5)):
        super().__init__(tag, device, model, dataset, poison_rate, lr, target_label, epochs, batch_size, optimizer, criterion, local_model_path)

        poisoned_train_data = create_poisoned_data(dataset)(self.data_path, self.transform, poison_rate, True, target_label, trigger_path, trigger_size)
        poisoned_test_data = create_poisoned_data(dataset)(self.data_path, self.transform, poison_rate, False, target_label, trigger_path, trigger_size)
        self.dataloader_train = DataLoader(poisoned_train_data, batch_size=self.batch_size, shuffle=True)
        self.dataloader_cleantest = DataLoader(self.clean_testdata, batch_size=self.batch_size, shuffle=True)
        self.dataloader_poisonedtest = DataLoader(poisoned_test_data, batch_size=self.batch_size, shuffle=True)

    def train(self):
        print("Training on {",self.device,"}")
        min_loss = np.inf
        for epoch in range(self.epochs):
            train_stats = self.train_one_epoch(self.dataloader_train, self.model, self.optimizer, self.criterion, self.device)
            test_stats = self.evaluate_model(self.dataloader_cleantest, self.dataloader_poisonedtest, self.model, self.criterion, self.device)
            print(f"EPOCH {epoch + 1}/{self.epochs}   loss: {train_stats['loss']:.4f} CDA: {test_stats['CDA']:.4f}, ASR: {test_stats['ASR']:.4f}\n")
            if train_stats['loss'] < min_loss:
                print("Model updated ---",test_stats)
                torch.save(self.model, self.model_path)
                min_loss = train_stats['loss']

    def test(self):
        test_stats = self.evaluate_model(self.dataloader_cleantest, self.dataloader_poisonedtest, self.model, self.criterion, self.device)
        print(f"CDA: {test_stats['CDA']:.4f}, ASR: {test_stats['ASR']:.4f}\n")

    def display(self):
        for batch in self.dataloader_poisonedtest:
            one_img=np.transpose(self.detransform(batch[0][0]).numpy(), (1, 2, 0))
            plt.imshow(one_img)
            plt.axis('off')
            plt.show()
            return


