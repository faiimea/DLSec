from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as transforms
import time
import torch
import os
import numpy as np
from dataset import split_dataset,add_predefined_trigger_cifar,add_trigger_cifar,generate_trigger
from batchnorm import transfer_bn_to_noisy_bn
from anp_utils import mask_train,clip_mask,save_mask_scores,test,reset,evaluate_by_number,evaluate_by_threshold,read_data
from torchvision.datasets import CIFAR10
from ANP import ANP
'''
下面这些参数是需要调整的超参数
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_dir = r"Backdoor\Defense\ANP\Data\\"
output_dir = r"Backdoor\Defense\ANP\Data\\"
val_frac=0.01
batch_size=128
print_every=500
nb_iter = 500
anp_eps=0.02
anp_steps=1
anp_alpha=0.2
pruning_by='threshold'
# pruning_by='number'
pruning_max=0.90
pruning_step=0.05
poison_type='badnets'
poison_rate=0.05

net=torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
checkpoint=r"Backdoor\LocalModels\20231229-161017-BadnetCIFAR10.pth"


def main():
    
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.

    orig_train = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    clean_train, clean_val = split_dataset(dataset=orig_train, val_frac=val_frac)
    clean_test = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
   
    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None}
    trigger_type = triggers[poison_type]
    
    poison_test,trigger_info=add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=poison_rate,
                                     poison_target=0, trigger_alpha=anp_alpha,path=r'Backdoor\\Attack\\triggers\\logo.png')
    

    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=print_every * batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)
    poison_test_loader = DataLoader(poison_test, batch_size=batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=0)

    anp_defense=ANP(net, device, checkpoint, clean_val_loader, clean_test_loader, poison_test_loader)
    anp_defense.optimize_mask(nb_iter=nb_iter, print_every=print_every)
    anp_defense.pruning()


if __name__ == '__main__':
    main()
