import torchvision
from torch.utils.data import DataLoader
from utils.transform import build_transform
from Defense.Friendly_noise import *
from utils.utils import *




demodataset = torchvision.datasets.CIFAR10("../data", train=True, download=True)
transform, detransform, channels = build_transform(demodataset.data.shape[1:], isinstance(demodataset.data[0], torch.Tensor))
demodataset.transform = transform
demodataloader = DataLoader(demodataset, batch_size=64, shuffle=False)

pretrained_model=torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

friendly_noise_params={
    'model':pretrained_model,
    'original_dataloader':demodataloader,
    'device':torch.device('cuda'),
    'friendly_epochs':30,
    'mu':1,
    'friendly_lr':0.1,
    'friendly_momentum':0.9,
    'clamp_min':-32/255,
    'clamp_max':32/255
}


reinforced_dataset=reinforce_dataset(path="./Friendly_noise_data/",tag="demoCIFAR10",load=False,**friendly_noise_params)
reinforced_dataloader=DataLoader(reinforced_dataset,batch_size=64,shuffle=True)

test_stats=evaluate(demodataloader,pretrained_model)
print(f"Original_CDA:{test_stats['acc']}\n")
test_stats=evaluate(reinforced_dataloader,pretrained_model)
print(f"Reinforced_CDA:{test_stats['acc']}\n")
