import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader
from Adversarial.adversarial_api import adversarial_attack
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.Defense import *
from utils.transform import build_transform
import copy
from EvaluationConfig import *
from Datapoison.Defense.Friendly_noise import *


def ModelEvaluation(evaluation_params=None):
    """
    @param params: 其它参数
    @param model: 待测模型
    @param adversarial_method:
    @param backdoor_method:
    @param datapoison_method:
    @return:
    """
    train_dataloader, test_dataloader = dataset_preprocess(name=evaluation_params['use_dataset'], batch_size=evaluation_params['batch_size'])
    isBackdoored, isPoisoned = run_test_on_model(evaluation_params['model'], evaluation_params['adversarial_method'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
                                                 evaluation_params['datapoison_reinforce_method'], train_dataloader, test_dataloader, evaluation_params)

    # run_test_on_model(ReinforcedModel, adversarial_method, backdoor_method, datapoison_method)


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

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
!!! HERE 注释掉了我不用的部分，后面补回来，再删掉这个注释
'''
def run_test_on_model(Model2BeEvaluated, adversarial_method, allow_backdoor_defense, backdoor_method, datapoison_method, run_datapoison_reinforcement, datapoison_reinforce_method, train_dataloader=None, test_dataloader=None, params=None):
    adversarial_rst = adversarial_test(Model2BeEvaluated, adversarial_method, test_dataloader, params)
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = backdoor_detect_and_defense(allow_defense=allow_backdoor_defense, Model2BeEvaluated=Model2BeEvaluated, method=backdoor_method, train_dataloader=train_dataloader, params=params)
    isPoisoned, datapoison_test_rst = datapoison_test(Model2BeEvaluated=Model2BeEvaluated, method=None, train_dataloader=train_dataloader, params=None)
    if run_datapoison_reinforcement:
        if ReinforcedModel_dict_path is not None:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
            DatapoisonReinforceModel.load_state_dict(torch.load(ReinforcedModel_dict_path))
        else:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
        datapoison_defense_rst, ReinforcedModel_dict_path = datapoison_defense(TargetModel=DatapoisonReinforceModel, method=datapoison_reinforce_method, train_dataloader=train_dataloader, params=params)
    else:
        datapoison_defense_rst = None

    process_result(adversarial_rst, backdoor_rst, datapoison_test_rst, datapoison_defense_rst)
    return isBackdoored, isPoisoned

'''
在此调用对抗攻击测试方法，传入待测模型、攻击方式、数据集（如果指定了）等用户设置的参数，返回测试结果：【原始与对抗样本准确率、准确率差，准确率降低一定比例（如50%）时对抗样本与原始样本差异度（p阶范数距离），】
进阶指标：【对抗样本对噪声容忍度，高斯模糊鲁棒性，图像压缩鲁棒性，生成对抗样本所需时间】
'''
def adversarial_test(Model2BeEvaluated, method='fgsm', train_dataloader=None, params=None):
    adversarial_rst = None
    adversarial_attack(Model2BeEvaluated,method,train_dataloader,params)
    return adversarial_rst


def backdoor_detect_and_defense(allow_defense=True, Model2BeEvaluated=None, method='NeuralCleanse', train_dataloader=None, params=None):
    print("开始后门检测")
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = run_backdoor_defense(allow_defense, Model2BeEvaluated, method, train_dataloader, params)
    print("-" * 20, "后门攻击评测结果", "-" * 20)
    print("后门检测：", end="")
    if isBackdoored:
        print("检测到后门存在于标签", backdoor_rst)
        print("防御后模型已存入目录", ReinforcedModel_dict_path)
    else:
        print("未检测到后门")
    return isBackdoored, backdoor_rst, ReinforcedModel_dict_path


def datapoison_test(Model2BeEvaluated=None, method=None, train_dataloader=None, params=None):
    # 在此执行数据投毒测试，如果run_defense为True则生成加强模型

    return None, None


def datapoison_defense(TargetModel=None, method=None, train_dataloader=None, params=None):

    return None, None


def process_result(adversarial_rst, backdoor_rst, datapoison_rst, datapoison_defense_rst):
    return


if __name__ == '__main__':
    ModelEvaluation(evaluation_params=evaluation_params)
