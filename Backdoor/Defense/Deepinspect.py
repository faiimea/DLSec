import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import Subset
import scipy.stats as stats
import copy
import os
from Backdoor.Defense.gen import Generator
'''
deepinspect检测的关键在于训练一个generator来生成后门，
这里由于生成后门的网络和数据集shape有关，下面写了针对cifar10和mnist的结构，
因此对于一个模型会有对应一个gen，检测时也是产生所有可能的后门，检查左侧离群值
主要处理过程如下
    数据集和gen的处理，数据集一部分用于train一部分用于eval，我写的比较丑陋
    训练gen或者直接读取
    后门检测，得到可能的后门对应标签的列表
    后门防御
'''


# class Generator(nn.Module):
#     """cifar数据集"""
#
#     def __init__(self, label_num=10, channels=3, height=32, width=32):
#         super(Generator, self).__init__()
#         self.x, self.y = int(height / 4), int(width / 4)
#         self.k1, self.k2 = height % 4, width % 4
#         if self.k1 == 0:
#             self.k1, self.x = 4, self.x - 1
#         if self.k2 == 0:
#             self.k2, self.y = 4, self.y - 1
#         self.linear1 = nn.Linear(label_num, 128 * self.x * self.y)
#         self.bn1 = nn.BatchNorm1d(128 * self.x * self.y)
#         self.linear2 = nn.Linear(100, 128 * self.x * self.y)
#         self.bn2 = nn.BatchNorm1d(128 * self.x * self.y)
#         self.deconv1 = nn.ConvTranspose2d(256, 128,
#                                           kernel_size=(4, 4),
#                                           padding=1)
#         '''output=(input-1)*stride-2*padding+kernel_size+(output+2*padding-kernel_size)%stride'''
#         self.bn3 = nn.BatchNorm2d(128)
#         self.deconv2 = nn.ConvTranspose2d(128, 64,
#                                           kernel_size=(4, 4),
#                                           stride=2,
#                                           padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.deconv3 = nn.ConvTranspose2d(64, channels,
#                                           kernel_size=(self.k1, self.k2),
#                                           stride=2,
#                                           padding=1)
#
#     def forward(self, x1, x2):
#         x1 = F.relu(self.linear1(x1))
#         x1 = self.bn1(x1)
#         x1 = x1.view(-1, 128, self.x, self.y)
#         x2 = F.relu(self.linear2(x2))
#         x2 = self.bn2(x2)
#         x2 = x2.view(-1, 128, self.x, self.y)
#         x = torch.cat([x1, x2], axis=1)
#         x = F.relu(self.deconv1(x))
#         x = self.bn3(x)
#         x = F.relu(self.deconv2(x))
#         x = self.bn4(x)
#         x = torch.tanh(self.deconv3(x))
#         return x


def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]


def test_gen_backdoor(gen, model, source_loader, target_label, device,num_class):
    gen.eval()
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, (img, ori_label) in enumerate(source_loader):
            label = torch.ones_like(ori_label) * target_label
            one_hot_label = one_hot(label,class_count=num_class).to(device)
            img, label = img.to(device), label.to(device)
            noise = torch.randn((img.shape[0], 100)).to(device)
            G_out = gen(one_hot_label, noise)
            D_out = model(img + G_out)
            pred = D_out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc = total_correct / total_count
    # save_image(G_out[0], '../Gen_trigger.png')
    return acc.item()


def test_clean(model, source_loader, device):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(source_loader):
            img, label = img.to(device), label.to(device)
            D_out = model(img)
            pred = D_out.data.max(1)[1]
            total_correct += pred.eq(label.data.view_as(pred)).sum()
            total_count += img.shape[0]
    acc = total_correct / total_count
    return acc.item()


def detect_triger(gen, device, alpha=0.05,num_class=10):
    noise = torch.randn((100, 100)).to(device)
    trigger_perturbations = []
    for target_class in range(num_class):
        label = torch.ones(100, dtype=torch.int64) * target_class
        one_hot_label = one_hot(label,class_count=num_class).to(device)
        G_out = gen(one_hot_label, noise).detach().cpu()
        abs_sum = torch.sum(torch.abs(G_out.view(G_out.shape[0], -1)), dim=1)
        value, index = torch.min(abs_sum, dim=0)
        trigger_perturbations.append(value.item())

    relative_size = G_out[0].shape[0] * 25
    # 计算触发器扰动的中位数
    median = np.median(trigger_perturbations)
    left_subgroup = [(i, median-x) for i, x in enumerate(trigger_perturbations) if x < median]
    # 计算左子组中数据点与组中位数的绝对偏差1
    median_1 = np.median([item[1] for item in left_subgroup])
    # 计算触发器扰动的标准差估计值
    mad = 1.4826 * np.median(np.abs([item[1] for item in left_subgroup] - median_1), axis=0)
    result= stats.norm.cdf((median-trigger_perturbations-median_1)/mad)

    other_result=result*relative_size/trigger_perturbations
    # 计算假设测试的显著性水平（α）alpha

    # 根据显著性水平计算截断阈值（c）
    

    # 根据 DMAD 检测标准，判断是否存在异常值
    outliers = np.where(result>1-alpha)[0]
    if len(outliers) > 0 and other_result[outliers]>0.8:
        print("存在异常值！", outliers)
        return outliers,[np.max(result),np.max(relative_size/np.array(trigger_perturbations))]
    else:
        print("未检测到异常值。")
        return None,[np.max(result),np.max(relative_size/np.array(trigger_perturbations))]


def train_gen(gen, model, epoch, dataloader, device, threshold=20, generator_path=None,num_class=10):
    model.eval()
    patience = 30
    noimpovement = 0
    bestloss = float("inf")

    for _ in range(1, epoch + 1):
        gen.train()
        optimizer = torch.optim.Adam(gen.parameters(), lr=1e-2)
        lamda1 = 0.6
        NLLLoss = nn.NLLLoss(reduction='sum')
        logsoftmax = nn.LogSoftmax(dim=1)
        Loss_sum = 0
        L_trigger_sum = 0
        L_pert_sum = 0
        count_sum = 0
        for i, (img, ori_label) in enumerate(dataloader):
            label = torch.randint(low=0, high=num_class, size=(img.shape[0],))
            one_hot_label = one_hot(label,class_count=num_class).to(device)
            img, label = img.to(device), label.to(device)
            noise = torch.randn((img.shape[0], 100)).to(device)
            G_out = gen(one_hot_label, noise)
            D_out = model(img + G_out)

            D_out = logsoftmax(D_out)
            L_trigger = NLLLoss(D_out, label)
            G_out_norm = torch.sum(torch.abs(G_out)) / img.shape[0] - threshold
            L_pert = torch.max(torch.zeros_like(G_out_norm), G_out_norm)

            Loss = L_trigger + lamda1 * L_pert
            optimizer.zero_grad()
            if Loss < bestloss:
                bestloss = Loss
            else:
                noimpovement += 1
            if noimpovement > patience:
                noimpovement = 0
                patience *= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
            Loss.backward()
            optimizer.step()
            Loss_sum += Loss.item()
            L_trigger_sum += L_trigger.item()
            L_pert_sum += L_pert.item()
            count_sum += 1
        # print(f'Epoch-{_}: Loss={round(Loss_sum / count_sum, 3)}, L_trigger={round(L_trigger_sum / count_sum, 3)}, L_pert={round(L_pert_sum / count_sum, 3)}, L_Gan={round(L_Gan_sum / count_sum, 3)}')
        print(f'Epoch-{_}: Loss={round(Loss_sum / count_sum, 3)}, L_trigger={round(L_trigger_sum / count_sum, 3)}, L_pert={round(L_pert_sum / count_sum, 3)}')


def deepinspect(model, train_dataloader, tag, generator_path=None, load_generator=False):
    clean_budget = 2000
    patch_rate = 0.15
    
    all_dataset = train_dataloader.dataset
    indices = np.random.choice(len(all_dataset), clean_budget, replace=False)
    dataset = Subset(all_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    labels = []
    for data in dataset:
        _, label = data  # 假设数据集对象中的每个样本包含数据和标签，且标签位于索引 1 处
        labels.append(label)

    # 统计标签种类数
    unique_labels = torch.unique(torch.tensor(labels))
    num_classes = len(unique_labels)
    device = "cuda"
    epoch = 20

    model.to(device)
    gen = Generator(channels=all_dataset[0][0].shape[0], height=all_dataset[0][0].shape[1], width=all_dataset[0][0].shape[2]).to(device)
    model.eval()
    '''
        如果之前训练好了gen这次只要检测或者防御可直接读取
    '''
    if load_generator:
        gen.load_state_dict(torch.load(generator_path))
    else:
        train_gen(gen=gen, epoch=epoch, model=model, dataloader=dataloader, device=device, generator_path=generator_path,num_class=num_classes)
    '''
    后门检测部分
    '''
    gen.eval()
    outliers,trigger = detect_triger(gen=gen, device=device,num_class=num_classes)
    if outliers is None:
        return [], trigger,None, None
    '''
    后面是防御部分
    '''
    newmodel = copy.deepcopy(model)
    noise = torch.randn((int(patch_rate * clean_budget), 100)).to(device)
    for target_class in outliers:
        testdataset = Subset(all_dataset, indices=[i for i in range(len(all_dataset)) if i not in indices and np.random.random() < 0.1])
        testdataloader = DataLoader(testdataset, batch_size=128, shuffle=True)
        bdacc1 = test_gen_backdoor(gen, newmodel, testdataloader, target_class, device,num_class=num_classes)
        cda1 = test_clean(newmodel, testdataloader, device)
        label = torch.ones((int(patch_rate * clean_budget)), dtype=torch.int64) * target_class
        one_hot_label = one_hot(label,class_count=num_classes).to(device)
        G_out = gen(one_hot_label, noise).detach().cpu()
        patched_dataset = []
        for i, (img, label) in enumerate(dataset):
            if i < int(patch_rate * clean_budget):
                img = img + G_out[i]
            patched_dataset.append((img, label))

        patched_loader = DataLoader(patched_dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.SGD(newmodel.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(20):
            newmodel.train()
            for i, (img, label) in enumerate(patched_loader):
                img, label = img.to(device), label.to(device)
                out = newmodel(img)
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bdacc2 = test_gen_backdoor(gen, newmodel, testdataloader, target_class, device,num_class=num_classes)
        cda2 = test_clean(newmodel, testdataloader, device)
        print("防御前的分类准确率：{:.2f}%".format(cda1 * 100))
        print("防御前的后门准确率：{:.2f}%".format(bdacc1 * 100))
        print("防御后的分类准确率：{:.2f}%".format(cda2 * 100))
        print("防御后的后门准确率：{:.2f}%".format(bdacc2 * 100))
    new_model_save_path = os.getcwd() + "/Backdoor/Defense/DeepInspectResult/"+ tag
    torch.save(newmodel.state_dict(), new_model_save_path)
    return outliers, trigger,newmodel, new_model_save_path
