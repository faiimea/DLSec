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
'''
下面这些参数是需要调整的超参数
'''
class ANP():
    def __init__(
            self, net, device, checkpoint,clean_val_loader,clean_test_loader,poison_test_loader,path=r'/Backdoor/Defense/ANP/Data/',
            anp_eps=0.2, anp_steps=1, anp_alpha=0.2, pruning_by='threshold', pruning_max=0.90, pruning_step=0.05):
        self.net = net
        self.device = device
        self.checkpoint = checkpoint
        self.anp_eps = anp_eps
        self.anp_steps = anp_steps
        self.anp_alpha = anp_alpha
        self.pruning_by = pruning_by
        self.pruning_max = pruning_max
        self.pruning_step = pruning_step
        self.clean_val_loader = clean_val_loader
        self.clean_test_loader = clean_test_loader
        self.poison_test_loader = poison_test_loader
        if not os.path.exists(path):
            os.makedirs(path)
        self.output_dir = path

    def optimize_mask(self,nb_iter=2000, print_every=500):
        '''使用mask_train函数训练mask，通过对神经元添加扰动'''
        net = self.net
        device = self.device
        net.load_state_dict(torch.load(self.checkpoint))
        net = transfer_bn_to_noisy_bn(net)
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=0.02, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.anp_eps / self.anp_steps)

        print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(nb_iter / print_every))
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=self.clean_val_loader,
                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer,device=device,anp_steps=self.anp_steps,anp_eps=self.anp_eps,anp_alpha=self.anp_alpha)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=self.clean_test_loader,device=device)
            po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=self.poison_test_loader,device=device)
            end = time.time()
            print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))
        save_mask_scores(net.state_dict(), os.path.join(self.output_dir, 'mask_values.txt'))
        print(os.path.join(self.output_dir, 'mask_values.txt'))

    def pruning(self,type=['threshold','number'],output_dir=r'/Backdoor/Defense/ANP/Data/'):
        '''通过对神经元剪枝实现防御'''
        net = self.net
        net=net.to(self.device)
        device = self.device
        mask_file = os.path.join(self.output_dir, 'mask_values.txt')
        criterion = torch.nn.CrossEntropyLoss().to(device)
        mask_values = read_data(mask_file)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=self.clean_test_loader,device=device)
        po_loss, po_acc = test(model=net, criterion=criterion, data_loader=self.poison_test_loader,device=device)
        print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
        if self.pruning_by == 'threshold':
            results = evaluate_by_threshold(
                net, mask_values, pruning_max=self.pruning_max, pruning_step=self.pruning_step,
                criterion=criterion, clean_loader=self.clean_test_loader, poison_loader=self.poison_test_loader,device=device
            )
        else:
            results = evaluate_by_number(
                net, mask_values, pruning_max=self.pruning_max, pruning_step=self.pruning_step,
                criterion=criterion, clean_loader=self.clean_test_loader, poison_loader=self.poison_test_loader,device=device)
        file_name = os.path.join(self.output_dir, 'pruning_by_{}.txt'.format(self.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)

if __name__ == '__main__':
    pass
