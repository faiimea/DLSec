import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
from Defense.base import BackdoorDefense
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
local_model_path="./checkpoints/20231202-002425-WaNetCifar10pretrained.pth"
print("Loading local model from path:", local_model_path)
model=torch.load(local_model_path)
model.to("cuda")
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
# 创建用于加载CIFAR10数据集的transform和dataloader
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std),
         ])
traindataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
traindataloader = DataLoader(traindataset, batch_size=64, shuffle=True)
"""
给出数据集的dataloader，黑盒模型，以及逆向后文件存储路径，默认在Defense下根据triggerpath创建的子文件夹
如果之前进行过逆向检测，那么run只显示之前计算的结果，否则会重新进行计算。
"""
bdd=BackdoorDefense(dataloader=traindataloader,model=model,triggerpath="20231202-002425-WaNetCifar10pretrained.pth")
bdd.run(alreadyreverse=False)