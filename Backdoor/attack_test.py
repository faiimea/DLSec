import torchvision.datasets
import torch
from Attack.BadNets import Badnets
from Attack.WaNet import WaNet
from Attack.blend import Blend
# import pretrainedmodels
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
# model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
# model=Net(1,10)
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)

'''
model=models.resnet50(pretrained=False)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100),
                            nn.LogSoftmax(dim=1))
'''
from Backdoor.Defense.Deepinspect import deepinspect
from EvaluationPlatformNEW import dataset_preprocess


test1229,_=dataset_preprocess("CIFAR10",128)


if __name__ == "__main__":
    attack_mode="Badnets"
    if attack_mode=="Badnets":
        Badnets_params = {
            'tag': "BadnetCIFAR10forVGG16",
            'device': 'cuda',
            'model': model,
            'dataset': torchvision.datasets.CIFAR10,
            'poison_rate': 0.1,
            'lr': 0.05,
            'target_label': 3,
            'epochs': 20,
            'batch_size': 128,
            'optimizer': 'sgd',
            'criterion': torch.nn.CrossEntropyLoss(),
            'local_state_path':None,  # LocalModels下相对路径
            'trigger_path': './Attack/triggers/trigger_10.png',
            'trigger_size': (5, 5)
        }
        badnet_victim = Badnets(**Badnets_params)
        # badnet_victim.train()
        badnet_victim.test()
        badnet_victim.display()


    elif attack_mode=="WaNet":
        WaNet_params = {
            'tag': "WaNetCIFAR",
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
            'local_state_path': '20231222-121433-WaNetCIFAR.pth',  # LocalModels下相对路径
            's': 0.5,
            'k': 4,
            'noise_ratio': 0.2
        }
        WaNet_victim = WaNet(**WaNet_params)
        # WaNet_victim.train()
        WaNet_victim.test()
        WaNet_victim.display()

    elif attack_mode=="Blend":
        Blend_params = {
            'tag': "BlendCIFAR10",
            'device': 'cuda',
            'model': model,
            'dataset': torchvision.datasets.CIFAR10,
            'poison_rate': 0.05,
            'lr': 0.05,
            'target_label': 3,
            'epochs': 30,
            'batch_size': 128,
            'optimizer': 'sgd',
            'criterion': torch.nn.CrossEntropyLoss(),
            'local_state_path': None,  # LocalModels下相对路径
            'blend_pic_path': './Attack/triggers/logo.png',
            'blend_ratio': 0.1
        }
        Blend_victim=Blend(**Blend_params)
        Blend_victim.train()
        Blend_victim.test()
        Blend_victim.display()
    else:
        raise NotImplementedError




