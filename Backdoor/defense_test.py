import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,MNIST
from torch.utils.data import DataLoader
import torch
from Defense.NeuralCleanse import NeuralCleanse
import numpy as np

'''
准备数据集和模型
NC初始化的参数X和Y是numpy的格式，这里要求是干净的训练集
NC也是分为三个步骤， 
    优化逆向后门过程，后门存储在trigger.npy文件中，对于确定的模型和训练集，逆向只需要进行一次
    后门检测：backdoor_detection()
    后门防御：mitigate()
实际上如果只需要检测对黑盒模型即可，防御需要是白盒，deepinspect亦然
'''
local_model_path="./Backdoor/LocalModels/20231229-161017-BadnetCIFAR10.pth"
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
print("Loading local model from path:", local_model_path)
model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cuda')))
model=model.to("cuda")
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
# 创建用于加载CIFAR10数据集的transform和dataloader
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std),
         ])
traindataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
traindataloader = DataLoader(traindataset, batch_size=64, shuffle=True)
org_img=[]
org_label=[]
for data in traindataset:
    image,label=data
    org_img.append(image.permute(1, 2, 0).numpy())
    org_label.append(int(label))
org_img=np.array(org_img)
org_label=np.array(org_label)
NC = NeuralCleanse(X=org_img, Y=org_label, model=model, num_samples=25,path='/badnet')
NC.reverse_engineer_triggers()
NC.backdoor_detection()


testset = CIFAR10(root='../data', train=False, download=False, transform=transform)
testX = []
testY = []
for data in testset:
    img, label = data
    testX.append(img.permute(1, 2, 0).numpy())
    testY.append(int(label))
NC.mitigate(test_X=testX, test_Y=testY)
torch.save(NC.model,"./LocalModels/NCbadnet.pth")