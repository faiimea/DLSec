from Backdoor.Attack.WaNet import WaNet
from Datapoison.Defense.Friendly_noise import reinforce_dataset
import torch
import torchvision
from torch.utils.data import DataLoader

model=torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

"""先用后门攻击做demo了，实际friendly_noise针对的是干净标签攻击，等其他算法部署后再做测试"""

WaNet_params = {
            'tag': "WaNetCIFAR",
            'device': 'cuda',
            'model': model,
            'dataset': torchvision.datasets.CIFAR10,
            'data_download_path':"../../data",
            'poison_rate': 0.1,
            'lr': 0.1,
            'target_label': 2,
            'epochs': 5,
            'batch_size': 128,
            'optimizer': 'sgd',
            'criterion': torch.nn.CrossEntropyLoss(),
            'local_model_path': None,  # LocalModels下相对路径
            's': 0.5,
            'k': 4,
            'noise_ratio': 0.2
        }
WaNet_victim = WaNet(**WaNet_params)

WaNet_victim.train(5)
WaNet_victim.test()


friendly_noise_params={
    'model':WaNet_victim.model,
    'original_dataloader':WaNet_victim.dataloader_train,
    'device':torch.device('cuda'),
    'friendly_epochs':30,
    'mu':1,
    'friendly_lr':0.1,
    'friendly_momentum':0.9,
    'clamp_min':-32/255,
    'clamp_max':32/255
}
reinforced_dataset=reinforce_dataset(path="./",tag="demoCIFAR10",load=False,**friendly_noise_params)
reinforced_dataloader=DataLoader(reinforced_dataset,batch_size=64,shuffle=True)

WaNet_victim.dataloader_train=reinforced_dataloader
WaNet_victim.train(20)
WaNet_victim.test()
