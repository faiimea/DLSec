import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from utils.utils import train_one_epoch,evaluate
from torch.utils.data import DataLoader

class ReinforcedDataset():
    def __init__(self, dataset, perturbation=None):
        self.dataset = dataset
        self.perturbation = perturbation

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        if self.perturbation[index] is not None:
            perturb = self.perturbation[index]
            img += perturb
            img = torch.clamp(img, -1, 1)

        return img, label

    def __len__(self):
        return len(self.dataset)


def friendly_loss(output, target, eps, criterion):
    emp_risk = criterion(output, target)
    constraint = torch.mean(torch.square(eps))
    return emp_risk, constraint


def gen_friendly_noise(model, original_dataloader, device=torch.device('cuda'), friendly_epochs=30, mu=1, friendly_lr=0.1, friendly_momentum=0.9, clamp_min=-32 / 255, clamp_max=32 / 255):
    """ 生成友好噪声，注意此处与原版实现中不一样的在于友好噪声加入的时机，为保证数据处理通用性，此处将友好噪声叠加在归一化后的数据上，而非直接作用于原始图像

    @param model: 生成友好噪声所用基准模型
    @param original_dataloader:原始数据集
    @param device: 计算设备
    @param friendly_epochs:生成友好噪声训练轮数
    @param mu: 优化噪声时相似度与噪声大小的权重，mu越大越倾向于生成高噪声
    @param friendly_lr: 生成友好噪声学习率
    @param friendly_momentum: sgd优化器惯量
    @param clamp_min: 归一化友好噪声下限
    @param clamp_max: 归一化友好噪声上限
    @return: 与数据加载器同形状的噪声
    """
    nesterov = True
    model.to(device)
    model.eval()

    dataset_size = len(original_dataloader.dataset)
    friendly_noise = torch.zeros((dataset_size,) + original_dataloader.dataset[0][0].shape)

    noise_idx = 0
    for batch_idx, (inputs, target) in enumerate(tqdm(original_dataloader)):
        inputs = inputs.to(device)
        init = (torch.rand(*(inputs.shape)) - 0.5) * 2 * clamp_max
        eps = init.clone().detach().to(device).requires_grad_(True)

        optimizer = torch.optim.SGD([eps], lr=friendly_lr, momentum=friendly_momentum, nesterov=nesterov)

        # if friendly_steps is None:
        #     friendly_steps = [friendly_epochs // 2, friendly_epochs // 4 * 3]
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, friendly_steps)

        output_original = model(inputs)
        output_original = F.log_softmax(output_original, dim=1).detach()

        for friendly_epoch in range(friendly_epochs):
            eps_clamp = torch.clamp(eps, clamp_min, clamp_max)
            perturbed = torch.clamp(inputs + eps_clamp, -1, 1)

            output_perturb = model(perturbed)
            output_perturb = F.log_softmax(output_perturb, dim=1)

            emp_risk, constraint = friendly_loss(output_perturb, output_original, eps_clamp, torch.nn.KLDivLoss(reduction='batchmean', log_target=True))
            loss = emp_risk - mu * constraint

            optimizer.zero_grad()
            loss.backward()
            model.zero_grad()
            optimizer.step()

            # print(f"Friendly noise gen --- Batch {batch_idx} / {len(original_dataloader)}  "
            #       f"Epoch: {friendly_epoch}  -- Max: {torch.max(eps):.5f}  Min: {torch.min(eps):.5f}  "
            #       f"Mean (abs): {torch.abs(eps).mean():.5f}  Mean: {torch.mean(eps):.5f}  "
            #       f"Mean (abs) Clamp: {torch.abs(eps_clamp).mean():.5f}  Mean Clamp: {torch.mean(eps_clamp):.5f}  "
            #       f"emp_risk: {emp_risk:.3f}  constraint: {constraint:.3f}", end='\r', flush=True)

        friendly_noise[noise_idx:(noise_idx + inputs.shape[0])] = eps.cpu().detach()
        noise_idx += inputs.shape[0]

    friendly_noise = torch.clamp(friendly_noise, clamp_min, clamp_max)
    return friendly_noise


def reinforce_dataset(model=None, dataloader=None, path: str = "./Friendly_noise_data/", tag: str = None, load: bool = False, params=None):
    """ 将原始数据转变为带友好噪声的数据集

    @param params: 全局参数，同时会利用到其中定义的friendly_noise相关的参数，位于params['FRIENDLYNOISE_extra_config']
    @param path: 所用路径，当load为True时此处为noise数据的路径，当load为False时此处为存放noise数据的文件夹的路径
    @param tag: 标记此次生成的noise数据
    @param load: 是否加载已有的noise数据
    @param friendly_noise_params: 生成friendly noise所需相关参数
    @return: 一个加噪数据集
    """
    if load:
        friendly_noise = np.load(path)
    else:
        friendly_noise = gen_friendly_noise(model=model, original_dataloader=dataloader, device=params['device'],friendly_epochs=params['FRIENDLYNOISE_extra_config']['friendly_epochs'],mu=params['FRIENDLYNOISE_extra_config']['mu'],friendly_lr=params['FRIENDLYNOISE_extra_config']['friendly_lr'],friendly_momentum=params['FRIENDLYNOISE_extra_config']['friendly_momentum'],clamp_min=params['FRIENDLYNOISE_extra_config']['clamp_min'],clamp_max=params['FRIENDLYNOISE_extra_config']['clamp_max'])
        np.save(path + datetime.now().strftime("%Y%m%d-%H%M%S-" + tag), friendly_noise.numpy())

    friendly_noise_dataset = ReinforcedDataset(dataloader.dataset, friendly_noise)
    return friendly_noise_dataset


def datapoison_model_reinforce(model=None, dataloader=None, params=None):
    friendly_noise_dataset = reinforce_dataset(model=model, dataloader=dataloader, path=params['FRIENDLYNOISE_extra_config']['path'], tag=params['FRIENDLYNOISE_extra_config']['tag'], load=params['FRIENDLYNOISE_extra_config']['load'], params=params)
    friendly_noise_dataloader= DataLoader(friendly_noise_dataset, batch_size=params['FRIENDLYNOISE_extra_config']['train_batch_size'])

    min_loss=torch.inf
    best_acc=0
    for epoch in range(params['FRIENDLYNOISE_extra_config']['train_epochs']):
        train_stats = train_one_epoch(friendly_noise_dataloader, model, params['FRIENDLYNOISE_extra_config']['train_optimizer'](model.parameters(),lr=params['FRIENDLYNOISE_extra_config']['train_lr']), params['FRIENDLYNOISE_extra_config']['train_criterion'], params['device'])
        test_stats = evaluate(friendly_noise_dataloader, model, params['FRIENDLYNOISE_extra_config']['train_criterion'],params['device'])
        print(f"EPOCH {epoch + 1}/{params['FRIENDLYNOISE_extra_config']['train_epochs']}   loss: {train_stats['loss']:.4f} ACC: {test_stats['acc']:.4f}")
        if train_stats['loss'] < min_loss:
            print("Model updated ---", test_stats)
            torch.save(model.state_dict(), params['FRIENDLYNOISE_extra_config']['reinforced_model_path'])
            min_loss = train_stats['loss']
            best_acc=test_stats['acc']
    return params['FRIENDLYNOISE_extra_config']['reinforced_model_path'],best_acc
