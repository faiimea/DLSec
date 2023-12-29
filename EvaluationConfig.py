import torchvision.models as models
import torch

'''********** Load your model here **********'''
# model = models.resnet50(pretrained=True)
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False)
model.load_state_dict(torch.load('./Backdoor/LocalModels/20231229-161017-BadnetCIFAR10.pth'))

FRIENDLYNOISE_config = {
    'friendly_epochs': 30,
    'mu': 1,
    'friendly_lr': 0.1,
    'friendly_momentum': 0.9,
    'clamp_min': -32 / 255,
    'clamp_max': 32 / 255
}



evaluation_params = {
    'model': model,
    'adversarial_method': 'fgsm',
    'backdoor_method': 'DeepInspect',
    'allow_backdoor_defense': True,
    'datapoison_method': None,
    'datapoison_reinforce_method': 'FriendlyNoise',
    'run_datapoison_reinforcement': True,
    'use_dataset': 'CIFAR10',
    'batch_size': 64,
    'device': 'cuda',
    'tag': "resnet50",
    # 以下为部分方法会使用到的参数
    'DEEPINSPECT_generator_path': './Backdoor/Defense/DeepInspectResult/generator.pth',
    'DEEPINSPECT_load_generator': False,
    'FRIENDLYNOISE_extra_config':FRIENDLYNOISE_config,
}

