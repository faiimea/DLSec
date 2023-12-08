from torchvision import transforms
import torch


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
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        channels = 3
    elif len(mode) == 2:
        mean, std = (0.5,), (0.5,)
        channels = 1
        if isTensor:
            transform.transforms.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
    elif len(mode) == 3 and mode[2] == 4:
        mean, std = (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
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