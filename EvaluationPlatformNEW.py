import csv

import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader
from Adversarial.adversarial_api import adversarial_attack
from Adversarial.adversarial_api import adversarial_mutiple_attack
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.datapoison_api import *
from utils.transform import build_transform
import copy
from EvaluationConfig import *
from Datapoison.Defense.Friendly_noise import *


def ModelEvaluation(evaluation_params=None):
    train_dataloader, test_dataloader = dataset_preprocess(name=evaluation_params['use_dataset'], batch_size=evaluation_params['batch_size'])
    isBackdoored = run_test_on_model(evaluation_params['model'], evaluation_params['adversarial_method'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
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


def run_test_on_model(Model2BeEvaluated, adversarial_method, allow_backdoor_defense, backdoor_method, datapoison_method, run_datapoison_reinforcement, datapoison_reinforce_method, train_dataloader=None, test_dataloader=None, params=None):
    # adversarial_rst = adversarial_test(Model2BeEvaluated, adversarial_method, test_dataloader, params)
    isBackdoored, backdoor_rst,trigger, ReinforcedModel_dict_path = backdoor_detect_and_defense(allow_defense=allow_backdoor_defense, Model2BeEvaluated=Model2BeEvaluated, method=backdoor_method, train_dataloader=train_dataloader, params=params)
    # datapoison_test_rst = datapoison_test(Model2BeEvaluated=Model2BeEvaluated, method=datapoison_method, train_dataloader=train_dataloader, params=params)
    ReinforcedModel_dict_path = None
    if run_datapoison_reinforcement:
        if ReinforcedModel_dict_path is not None:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
            DatapoisonReinforceModel.load_state_dict(torch.load(ReinforcedModel_dict_path))
        else:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
        ReinforcedModel_dict_path, datapoison_defense_rst = datapoison_defense(TargetModel=DatapoisonReinforceModel, method=datapoison_reinforce_method, train_dataloader=train_dataloader, params=params)
    else:
        datapoison_defense_rst = None

    # process_result(params['tag'],adversarial_rst, backdoor_rst, datapoison_test_rst, datapoison_defense_rst)
    return isBackdoored


'''
在此调用对抗攻击测试方法，传入待测模型、攻击方式、数据集（如果指定了）等用户设置的参数，返回测试结果：【原始与对抗样本准确率、准确率差，准确率降低一定比例（如50%）时对抗样本与原始样本差异度（p阶范数距离），】
进阶指标：【对抗样本对噪声容忍度，高斯模糊鲁棒性，图像压缩鲁棒性，生成对抗样本所需时间】
'''


def adversarial_test(Model2BeEvaluated, method='fgsm', train_dataloader=None, params=None):
    """

    @param Model2BeEvaluated:
    @param method:
    @param train_dataloader:
    @param params:
    @return: 一个字典，键形如’ACC-0.005‘，’fgsm-0.005‘，值为相应准确率
    """
    adversarial_rst = {}
    print("开始图像鲁棒性检测")
    perturb_rst = adversarial_attack(Model2BeEvaluated, method, train_dataloader, params)
    for ep_rst in perturb_rst:
        ep = ep_rst[0]
        adversarial_rst["ACC-" + str(ep)] = ep_rst[1]
        adversarial_rst["NoisyACC-" + str(ep)] = ep_rst[2]
        adversarial_rst["BlurredACC-" + str(ep)] = ep_rst[3]
        adversarial_rst["CompressedACC-" + str(ep)] = ep_rst[4]

    print("开始多方法对抗样本测试")
    adv_rst = adversarial_mutiple_attack(Model2BeEvaluated, train_dataloader, params)
    for method_rst in adv_rst:
        ep = method_rst[0]
        adversarial_rst[str(method_rst[1]) + '-' + str(ep)] = method_rst[2]
    return adversarial_rst


def backdoor_detect_and_defense(allow_defense=True, Model2BeEvaluated=None, method='NeuralCleanse', train_dataloader=None, params=None):
    print("开始后门检测")
    isBackdoored, backdoor_rst, trigger,ReinforcedModel_dict_path = run_backdoor_defense(allow_defense, Model2BeEvaluated, method, train_dataloader, params)
    print("-" * 20, "后门攻击评测结果", "-" * 20)
    print("后门检测：", end="")
    if isBackdoored:
        print("检测到后门存在于标签", backdoor_rst)
        print("防御后模型已存入目录", ReinforcedModel_dict_path)
    else:
        print("未检测到后门")
    return isBackdoored, backdoor_rst, trigger,ReinforcedModel_dict_path


def datapoison_test(Model2BeEvaluated=None, method=None, train_dataloader=None, params=None):
    # 在此执行数据投毒测试，如果run_defense为True则生成加强模型
    # 返回投毒后概率
    print("开始投毒检测")
    datapoison_test_rst = run_datapoison_test(model=Model2BeEvaluated, method=method, train_dataloader=train_dataloader, params=params)

    return datapoison_test_rst


def datapoison_defense(TargetModel=None, method=None, train_dataloader=None, params=None):
    print("开始投毒防御")
    reinforced_model_path, datapoison_defense_rst = run_datapoison_reinforce(TargetModel, method=method, train_dataloader=train_dataloader, params=params)
    return reinforced_model_path, datapoison_defense_rst


def process_result(tag="DefaultTag", adversarial_rst=None, backdoor_rst=None, datapoison_rst=None, datapoison_defense_rst=None):
    print("Evaluation Results for", tag)
    print('adversarial_rst:', adversarial_rst)
    # {'ACC-0.005': 81.25, 'NoisyACC-0.005': 17.410714285714285, 'BlurredACC-0.005': 8.258928571428571, 'CompressedACC-0.005': 11.830357142857142, 'ACC-0.01': 78.34821428571429, 'NoisyACC-0.01': 13.616071428571429, 'BlurredACC-0.01': 10.491071428571429, 'CompressedACC-0.01': 8.928571428571429, 'fgsm-0.005': 84.375, 'pgd-0.005': 84.375, 'difgsm-0.005': 54.6875, 'mifgsm-0.005': 54.6875, 'nifgsm-0.005': 54.6875, 'sinifgsm-0.005': 54.6875, 'tifgsm-0.005': 61.71875, 'vmifgsm-0.005': 54.6875, 'vnifgsm-0.005': 56.25}
    print('backdoor_rst:', backdoor_rst)
    print('datapoison_rst:', datapoison_rst)
    print('datapoison_defense_rst:', datapoison_defense_rst)
    final_rst = {'tag':tag}
    if adversarial_rst is not None:
        final_rst.update(adversarial_rst)
    if backdoor_rst is not None:
        final_rst.update(backdoor_rst)
    if datapoison_rst is not None:
        final_rst.update(datapoison_rst)
    if datapoison_defense_rst is not None:
        final_rst.update(datapoison_defense_rst)

    with open('./ModelResults.csv', 'r', newline='') as csvfile:
        data=[]
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
        headers=set(data[0].keys())
        headers.update(final_rst.keys())
        data.append(final_rst)

    with open('./ModelResults.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    return


if __name__ == '__main__':
    ModelEvaluation(evaluation_params=evaluation_params)
import csv

import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader
from Adversarial.adversarial_api import adversarial_attack
from Adversarial.adversarial_api import adversarial_mutiple_attack
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.datapoison_api import *
from utils.transform import build_transform
import copy
from EvaluationConfig import *
from Datapoison.Defense.Friendly_noise import *


def ModelEvaluation(evaluation_params=None):
    train_dataloader, test_dataloader = dataset_preprocess(name=evaluation_params['use_dataset'], batch_size=evaluation_params['batch_size'])
    isBackdoored = run_test_on_model(evaluation_params['model'], evaluation_params['adversarial_method'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
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


def run_test_on_model(Model2BeEvaluated, adversarial_method, allow_backdoor_defense, backdoor_method, datapoison_method, run_datapoison_reinforcement, datapoison_reinforce_method, train_dataloader=None, test_dataloader=None, params=None):
    adversarial_rst = adversarial_test(Model2BeEvaluated, adversarial_method, test_dataloader, params)
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = backdoor_detect_and_defense(allow_defense=allow_backdoor_defense, Model2BeEvaluated=Model2BeEvaluated, method=backdoor_method, train_dataloader=train_dataloader, params=params)
    datapoison_test_rst = datapoison_test(Model2BeEvaluated=Model2BeEvaluated, method=datapoison_method, train_dataloader=train_dataloader, params=params)
    ReinforcedModel_dict_path = None
    if run_datapoison_reinforcement:
        if ReinforcedModel_dict_path is not None:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
            DatapoisonReinforceModel.load_state_dict(torch.load(ReinforcedModel_dict_path))
        else:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
        ReinforcedModel_dict_path, datapoison_defense_rst = datapoison_defense(TargetModel=DatapoisonReinforceModel, method=datapoison_reinforce_method, train_dataloader=train_dataloader, params=params)
    else:
        datapoison_defense_rst = None

    process_result(params['tag'],adversarial_rst, backdoor_rst, datapoison_test_rst, datapoison_defense_rst)
    return isBackdoored


'''
在此调用对抗攻击测试方法，传入待测模型、攻击方式、数据集（如果指定了）等用户设置的参数，返回测试结果：【原始与对抗样本准确率、准确率差，准确率降低一定比例（如50%）时对抗样本与原始样本差异度（p阶范数距离），】
进阶指标：【对抗样本对噪声容忍度，高斯模糊鲁棒性，图像压缩鲁棒性，生成对抗样本所需时间】
'''


def adversarial_test(Model2BeEvaluated, method='fgsm', train_dataloader=None, params=None):
    """

    @param Model2BeEvaluated:
    @param method:
    @param train_dataloader:
    @param params:
    @return: 一个字典，键形如’ACC-0.005‘，’fgsm-0.005‘，值为相应准确率
    """
    adversarial_rst = {}
    print("开始图像鲁棒性检测")
    perturb_rst = adversarial_attack(Model2BeEvaluated, method, train_dataloader, params)
    for ep_rst in perturb_rst:
        ep = ep_rst[0]
        adversarial_rst["ACC-" + str(ep)] = ep_rst[1]
        adversarial_rst["NoisyACC-" + str(ep)] = ep_rst[2]
        adversarial_rst["BlurredACC-" + str(ep)] = ep_rst[3]
        adversarial_rst["CompressedACC-" + str(ep)] = ep_rst[4]

    print("开始多方法对抗样本测试")
    adv_rst = adversarial_mutiple_attack(Model2BeEvaluated, train_dataloader, params)
    for method_rst in adv_rst:
        ep = method_rst[0]
        adversarial_rst[str(method_rst[1]) + '-' + str(ep)] = method_rst[2]
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
    # 返回投毒后概率
    print("开始投毒检测")
    datapoison_test_rst = run_datapoison_test(model=Model2BeEvaluated, method=method, train_dataloader=train_dataloader, params=params)

    return datapoison_test_rst


def datapoison_defense(TargetModel=None, method=None, train_dataloader=None, params=None):
    print("开始投毒防御")
    reinforced_model_path, datapoison_defense_rst = run_datapoison_reinforce(TargetModel, method=method, train_dataloader=train_dataloader, params=params)
    return reinforced_model_path, datapoison_defense_rst


def process_result(tag="DefaultTag", adversarial_rst=None, backdoor_rst=None, datapoison_rst=None, datapoison_defense_rst=None):
    print("Evaluation Results for", tag)
    print('adversarial_rst:', adversarial_rst)
    # {'ACC-0.005': 81.25, 'NoisyACC-0.005': 17.410714285714285, 'BlurredACC-0.005': 8.258928571428571, 'CompressedACC-0.005': 11.830357142857142, 'ACC-0.01': 78.34821428571429, 'NoisyACC-0.01': 13.616071428571429, 'BlurredACC-0.01': 10.491071428571429, 'CompressedACC-0.01': 8.928571428571429, 'fgsm-0.005': 84.375, 'pgd-0.005': 84.375, 'difgsm-0.005': 54.6875, 'mifgsm-0.005': 54.6875, 'nifgsm-0.005': 54.6875, 'sinifgsm-0.005': 54.6875, 'tifgsm-0.005': 61.71875, 'vmifgsm-0.005': 54.6875, 'vnifgsm-0.005': 56.25}
    print('backdoor_rst:', backdoor_rst)
    print('datapoison_rst:', datapoison_rst)
    print('datapoison_defense_rst:', datapoison_defense_rst)
    final_rst = {'tag':tag}
    if adversarial_rst is not None:
        final_rst.update(adversarial_rst)
    if backdoor_rst is not None:
        final_rst.update(backdoor_rst)
    if datapoison_rst is not None:
        final_rst.update(datapoison_rst)
    if datapoison_defense_rst is not None:
        final_rst.update(datapoison_defense_rst)

    with open('./ModelResults.csv', 'r', newline='') as csvfile:
        data=[]
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
        headers=set(data[0].keys())
        headers.update(final_rst.keys())
        data.append(final_rst)

    with open('./ModelResults.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    return


if __name__ == '__main__':
    ModelEvaluation(evaluation_params=evaluation_params)
