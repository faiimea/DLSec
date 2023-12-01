import torchvision.datasets
import torch
from Attack.BadNets import Badnets
from Attack.WaNet import WaNet
from LocalModels.net import Net
import pretrainedmodels
from torchvision import models
from torch import nn

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
# model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
'''
model=models.resnet50(pretrained=False)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100),
                            nn.LogSoftmax(dim=1))
'''

if __name__ == "__main__":
    attack_mode="WaNet"

    if attack_mode=="Badnets":
        Badnets_params = {
            'tag': "Badnetcifar10pretrained",
            'device': 'cuda',
            'model': model,
            'dataset': torchvision.datasets.CIFAR10,
            'poison_rate': 0.05,
            'lr': 0.05,
            'target_label': 3,
            'epochs': 20,
            'batch_size': 128,
            'optimizer': 'sgd',
            'criterion': torch.nn.CrossEntropyLoss(),
            'local_model_path': None,  # LocalModels下相对路径
            'trigger_path': './Attack/triggers/trigger_10.png',
            'trigger_size': (5, 5)
        }
        badnet_victim = Badnets(**Badnets_params)
        badnet_victim.train()
        # badnet_victim.test()
        # badnet_victim.display()

    elif attack_mode=="WaNet":
        WaNet_params = {
            'tag': "WaNetCifar10pretrained",
            'device': 'cuda',
            'model': model,
            'dataset': torchvision.datasets.CIFAR10,
            'poison_rate': 0.1,
            'lr': 0.1,
            'target_label': 2,
            'epochs': 20,
            'batch_size': 128,
            'optimizer': 'sgd',
            'criterion': torch.nn.CrossEntropyLoss(),
            'local_model_path': "20231202-002425-WaNetCifar10pretrained.pth",  # LocalModels下相对路径
            's': 0.5,
            'k': 4,
            'noise_ratio': 0.2
        }
        WaNet_victim = WaNet(**WaNet_params)
        WaNet_victim.train()
        # WaNet_victim.test()
        # WaNet_victim.display()
    else:
        raise NotImplementedError




