"""模型评测总接口
读取一个模型，输入相关必要参数：入数据格式，分类标签数...

执行以下测评步骤：
1.运行对抗攻击检测鲁棒性
2.运行后门检测
3.运行数据投毒检测
4.记录测评数据
5.如有需要，执行后门和数据投毒防御
6.防御后模型再测评并记录数据
7.综合评价并给出各项具体数据

需要检测的模型放在Model_to_be_Tested目录内，传入相对路径
使用数据集填写torchvision支持的数据集名或者自己的数据集文件夹路径，若使用自定义数据集，在指定的路径下应有train和test两个文件夹，从而加载出两个dataloader，否则只能加载出一个dataloader
"""

import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader

from Adv_Sample import *
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.Defense import *
from utils.transform import build_transform
import copy

evaluation_params = {
    'model_path': None,  # 填写Model_to_be_Tested文件夹下相对路径
    'adversarial_method': 'fgsm',
    'backdoor_method': 'NeuralCleanse',
    'datapoison_method': None,
    'use_dataset': 'CIFAR10'
}


def ModelEvaluation(model_path: str = None, adversarial_method: str = None, backdoor_method: str = None, datapoison_method: str = None, use_dataset: str = 'CIFAR10', batch_size: int = 64):
    """

    @param model_path: 待测模型的路径
    @param adversarial_method:
    @param backdoor_method:
    @param datapoison_method:
    @param use_dataset: 填入torchvision.dataset支持的数据集名称；或自定义数据集的根目录，自动检测此目录下是否有train和test文件夹，如有则分别读取，如无则读取为一个数据集
    @param batch_size: 可选参数，指定加载数据集时的批次大小
    @return:
    """
    train_dataloader, test_dataloader = dataset_preprocess(use_dataset)
    Model2BeEvaluated = torch.load(model_path)
    isBackdoored, isPoisoned = run_test_on_model(Model2BeEvaluated, adversarial_method, backdoor_method, datapoison_method, train_dataloader, test_dataloader)

    ReinforcedModel = copy.deepcopy(Model2BeEvaluated)
    if isBackdoored:
        backdoor_defense(ReinforcedModel)
    if isPoisoned:
        datapoison_defense(ReinforcedModel)
    run_test_on_model(ReinforcedModel, adversarial_method, backdoor_method, datapoison_method)


def dataset_preprocess(name, batch_size=64):
    if name is None:
        return None, None
    elif name in torchvision.datasets.__all__:
        selected_dataset = getattr(torchvision.datasets, name)
        train_dataset = selected_dataset("./data", train=True, download=True)
        test_dataset = selected_dataset("./data", train=False, download=True)
    else:
        if os.path.exists(name + "/train") and os.path.exists(name + "/test"):
            train_dataset = torchvision.datasets.DatasetFolder(root=name + "/train", loader=cv2.imread, extensions=('png', 'jpeg'))
            test_dataset = torchvision.datasets.DatasetFolder(root=name + "/test", loader=cv2.imread, extensions=('png', 'jpeg'))
        else:
            custom_dataset = torchvision.datasets.DatasetFolder(root=name, loader=cv2.imread, extensions=('png', 'jpeg'))
            transform, _, _ = build_transform(custom_dataset[0][0].shape, isinstance(custom_dataset[0], torch.Tensor))
            custom_dataset.transform = transform
            return DataLoader(custom_dataset, batch_size=batch_size, shuffle=True), None

    transform, _, _ = build_transform(train_dataset.data.shape[1:], isinstance(train_dataset.data[0], torch.Tensor))
    train_dataset.transform = transform
    test_dataset.transform = transform

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def run_test_on_model(model, adversarial_method, backdoor_method, datapoison_method, train_dataloader=None, test_dataloader=None):
    adversarial_rst = adversarial_test(model, adversarial_method, train_dataloader)
    isBackdoored, backdoor_rst = backdoor_detect(model, backdoor_method, train_dataloader)
    isPoisoned, datapoison_rst = datapoison_detect(model, datapoison_method, train_dataloader)
    process_result(adversarial_rst, backdoor_rst, datapoison_rst)
    return isBackdoored, isPoisoned


def adversarial_test(model, method='fgsm', train_dataloader=None):
    adversarial_rst = None
    '''在此调用对抗攻击测试方法，传入待测模型、攻击方式、数据集（如果指定了）等用户设置的参数，返回测试结果：【原始与对抗样本准确率、准确率差，准确率降低一定比例（如50%）时对抗样本与原始样本差异度（p阶范数距离），】
        进阶指标：【对抗样本对噪声容忍度，高斯模糊鲁棒性，图像压缩鲁棒性，生成对抗样本所需时间】
    '''
    return adversarial_rst


def backdoor_detect(model, method='NeuralCleanse', train_dataloader=None):
    # 在此执行后门检测
    isBackdoored = False
    backdoor_rst = None
    return isBackdoored, backdoor_rst


def datapoison_detect(model, method=None):
    # 在此执行数据投毒检测
    isPoisoned = False
    datapoison_rst = None
    return isPoisoned, datapoison_rst


def backdoor_defense(model):
    # 在此执行后门防御
    return


def datapoison_defense(model):
    # 在此执行数据投毒防御
    return


def process_result(adversarial_rst, backdoor_rst, datapoison_rst):
    return


if __name__ == "__main__":
    # ModelEvaluation(**evaluation_params)
    # r'C:\Users\Lenovo\Desktop\DLSec\data\ImageNet-Mini'
    print(dataset_preprocess('CIFAR10'))
