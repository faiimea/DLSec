from torchvision import transforms
import torch

"""
Imagenet:
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

CIFAR10:
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

CIFAR100:
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

SVHN:
mean = [0.4377, 0.4438, 0.4728]
std = [0.1980, 0.2010, 0.1970]

FashionMNIST:
mean  = [0.2860]
std = [0.3530]

MNIST:
mean = [0.1307]
std = [0.3081]

"""


def build_transform(mode, isTensor=False):
    """将图片转为归一化的张量

        mode: 原图片通道数
        isTensor:原图片是否本身就为张量
    """
    transform = transforms.Compose([])
    if not isTensor:
        transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(transforms.Lambda(to_float_tensor))
    if len(mode) == 3 and mode[2] == 3:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        channels = 3
    elif len(mode) == 2:
        mean, std = (0.1307,), (0.3081,)
        channels = 1
        if isTensor:
            transform.transforms.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
    elif len(mode) == 3 and mode[2] == 4:
        mean, std = (0.485, 0.456, 0.406, 0.5), (0.229, 0.224, 0.225, 0.5)
        channels = 4
    else:
        raise NotImplementedError()

    transform.transforms.append(transforms.Normalize(mean, std))

    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return transform, detransform, channels


def to_float_tensor(image):
    return image.float()