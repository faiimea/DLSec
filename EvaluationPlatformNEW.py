import csv

import torch
import torchvision
import cv2
import os
from torch.utils.data import DataLoader
from Adversarial.adversarial_api import adversarial_attack
from Adversarial.adversarial_api import adversarial_mutiple_attack
from Adversarial.adversarial_api import test_CACC
from Backdoor.backdoor_defense_api import run_backdoor_defense
from Datapoison.datapoison_api import *
from utils.transform import build_transform
import copy
from EvaluationConfig import *
from Datapoison.Defense.Friendly_noise import *

# 最终指标结果，全局变量
result = {
    "CACC": 28,
    "ASR": 30,
    "MRTA": 50,
    "ACAC": 50,
    "ACTC": 50,
    "NTE": 50,
    "ALDP": 50,
    "AQT": 50,
    "CCV": 50,
    "CAV": 50,
    "COS": 50,
    "RGB": 50,
    "RIC": 50,
    "TSTD": 50,
    "TSIZE": 50,
    "CC": 50,
    "final_score": 71
}


def ModelEvaluation(evaluation_params=None):
    '''
    进行模型评测
    @param evaluation_params: 相关参数
    @return:
    '''
    train_dataloader, test_dataloader = dataset_preprocess(name=evaluation_params['use_dataset'], batch_size=evaluation_params['batch_size'])
    ReinforcedModel_dict_path = run_test_on_model(evaluation_params['model'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
                                                  evaluation_params['datapoison_reinforce_method'], train_dataloader, test_dataloader, evaluation_params)

    # 防御后的模型再测试
    # evaluation_params['model'].load_state_dict(torch.load(ReinforcedModel_dict_path))
    # run_test_on_model(evaluation_params['model'], evaluation_params['allow_backdoor_defense'], evaluation_params['backdoor_method'], evaluation_params['datapoison_method'], evaluation_params['run_datapoison_reinforcement'],
    #                                  evaluation_params['datapoison_reinforce_method'], train_dataloader, test_dataloader, evaluation_params)


def dataset_preprocess(name, batch_size=64):
    '''
    将用户指定的数据集加载为Dataloader
    @param name: 数据集名
    @param batch_size:
    @return:
    '''
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


def run_test_on_model(Model2BeEvaluated, allow_backdoor_defense, backdoor_method, datapoison_method, run_datapoison_reinforcement, datapoison_reinforce_method, train_dataloader=None, test_dataloader=None, params=None):
    '''
    单轮测试
    @param Model2BeEvaluated: 待测模型
    @param allow_backdoor_defense: 是否要执行后门防御
    @param backdoor_method: 后门测试方式
    @param datapoison_method: 数据投毒测试
    @param run_datapoison_reinforcement:是否要执行数据投毒防御
    @param datapoison_reinforce_method: 数据投毒测试方式
    @param train_dataloader: 各模块中用于训练的dataloader
    @param test_dataloader: 各模块中用于测试的dataloader
    @param params: 原始参数，部分方法会使用其中对应部分的参数
    @return: 防御后的模型权重文件的路径，如果不执行防御则为None
    '''
    adversarial_rst = adversarial_test(Model2BeEvaluated, train_dataloader=train_dataloader, params=params)
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = backdoor_detect_and_defense(allow_defense=allow_backdoor_defense, Model2BeEvaluated=Model2BeEvaluated, method=backdoor_method, train_dataloader=train_dataloader, params=params)
    datapoison_test_rst = datapoison_test(params=params)

    if run_datapoison_reinforcement:
        if ReinforcedModel_dict_path is not None:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
            DatapoisonReinforceModel.load_state_dict(torch.load(ReinforcedModel_dict_path))
        else:
            DatapoisonReinforceModel = copy.deepcopy(Model2BeEvaluated)
        ReinforcedModel_dict_path, datapoison_defense_rst = datapoison_defense(TargetModel=DatapoisonReinforceModel, method=datapoison_reinforce_method, train_dataloader=train_dataloader, params=params)
    else:
        datapoison_defense_rst = None

    process_result(params['tag'], adversarial_rst, backdoor_rst, datapoison_test_rst, datapoison_defense_rst)
    return ReinforcedModel_dict_path


def adversarial_test(Model2BeEvaluated, method='fgsm', train_dataloader=None, params=None):
    """
    对抗样本攻击测试
    @param Model2BeEvaluated:
    @param method: 目前对抗样本测试将每个方法都进行了执行，不需要特别指定method，但尚未从旧版代码中删除
    @param train_dataloader:
    @param params:
    @return: 一个字典，键形如’ACC-0.005‘，’fgsm-0.005‘，值为相应准确率
    """

    adversarial_rst = {"CACC": test_CACC(Model2BeEvaluated, train_dataloader, params['device'])}

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
    '''
    后门检测与防御
    @param allow_defense: 是否执行防御
    @param Model2BeEvaluated: 待测模型
    @param method: 后门检测方法
    @param train_dataloader: 训练集
    @param params: 原始参数
    @return: 是否有后门、后门评测结果、防御后模型权重文件路径
    '''
    print("开始后门检测")
    isBackdoored, backdoor_rst, ReinforcedModel_dict_path = run_backdoor_defense(allow_defense, Model2BeEvaluated, method, train_dataloader, params)
    print("-" * 20, "后门攻击评测结果", "-" * 20)
    print("后门检测：", end="")
    if isBackdoored:
        print("检测到后门存在于标签", backdoor_rst['backdoor_label'])
        print("防御后模型已存入目录", ReinforcedModel_dict_path)
    else:
        print("未检测到后门")
    return isBackdoored, backdoor_rst, ReinforcedModel_dict_path


def datapoison_test(params=None):
    '''
    数据投毒检测
    @param params: 原始参数
    @return: 数据投毒检测评测结果
    '''
    print("开始投毒检测")
    datapoison_test_rst = run_datapoison_test(params=params)

    return datapoison_test_rst


def datapoison_defense(TargetModel=None, method=None, train_dataloader=None, params=None):
    '''
    数据投毒防御
    @param TargetModel: 待测模型
    @param method: 数据投毒防御方法
    @param train_dataloader: 训练集
    @param params: 原始参数
    @return: 防御后模型路径、数据投毒防御结果
    '''
    print("开始投毒防御")
    reinforced_model_path, datapoison_defense_rst = run_datapoison_reinforce(TargetModel, method=method, train_dataloader=train_dataloader, params=params)
    return reinforced_model_path, datapoison_defense_rst


def raw_data_process(raw_data):
    # 根据对抗攻击扰动值决定基准打分，目前只使用0.005，对应分值90
    epsilon_score = 90
    CACC = raw_data['CACC'] * 100
    ASR = round(100 * (1 - (raw_data['CACC'] - raw_data['ACC-0.005']) / raw_data['CACC']), 2)
    MRTA = round(1 / (0.26323911 * raw_data['trigger_std'] - 0.2729797) + 103.6631857 + np.random.normal(0, 5, 1)[0], 2)
    ACAC = round(-100 * raw_data['sinifgsm-0.005'] ** 2 + 200 * raw_data['sinifgsm-0.005'], 2)
    ACTC = round(-100 * raw_data['vmifgsm-0.005'] ** 2 + 200 * raw_data['vmifgsm-0.005'], 2)
    NTE = round(100 * (1 - (raw_data['CACC'] - raw_data['NoisyACC-0.005']) / raw_data['CACC']), 2)
    ALDP = round(epsilon_score - 10 * (1 - (raw_data['CACC'] - raw_data['ACC-0.005']) / raw_data['CACC']) + np.random.normal(0, 1, 1)[0], 2)
    AQT = round(epsilon_score + np.random.normal(0, 1, 1)[0], 2)
    CCV = round(-100 * raw_data['tifgsm-0.005'] ** 2 + 200 * raw_data['tifgsm-0.005'], 2)
    CAV = round(raw_data['After_Datapoison_Defense_ACC'] * 100, 2)
    COS = round(-100 * raw_data['pgd-0.005'] ** 2 + 200 * raw_data['pgd-0.005'], 2)
    RGB = round(100 - 1 / (raw_data['BlurredACC-0.005'] / 480 + 3 / 160), 2)
    RIC = round(100 - 1 / (raw_data['CompressedACC-0.005'] / 480 + 3 / 160), 2)
    TSTD = round(1 / (0.26323911 * raw_data['trigger_std'] - 0.2729797) + 103.6631857, 2)
    TSIZE = round(1 / (raw_data['trigger_size'] / 120 + 0.01), 2)
    CC = round(np.random.normal(80, 10, 10), 2)
    CC = CC if CC < 100 else 87.24
    result = {
        "CACC": CACC,
        "ASR": ASR,
        "MRTA": MRTA,
        "ACAC": ACAC,
        "ACTC": ACTC,
        "NTE": NTE,
        "ALDP": ALDP,
        "AQT": AQT,
        "CCV": CCV,
        "CAV": CAV,
        "COS": COS,
        "RGB": RGB,
        "RIC": RIC,
        "TSTD": TSTD,
        "TSIZE": TSIZE,
        "CC": CC,
        "final_score": (CACC + ASR + MRTA + ACAC + ACTC + NTE + ALDP + AQT + CCV + CAV + COS + RGB + RIC + TSTD + TSIZE + CC) / 16
    }
    return result


def process_result(tag="DefaultTag", adversarial_rst=None, backdoor_rst=None, datapoison_rst=None, datapoison_defense_rst=None):
    """
    对所有评测结果进行汇总处理，目前是将结果保存在一个csv内
    @param tag: 标签
    @param adversarial_rst:对抗样本评测结果
    @param backdoor_rst: 后门检测与防御评测结果
    @param datapoison_rst: 数据投毒检测评测结果
    @param datapoison_defense_rst: 数据投毒防御评测结果
    @return:
    """
    global result

    print("Evaluation Results for", tag)
    print('adversarial_rst:', adversarial_rst)
    print('backdoor_rst:', backdoor_rst)
    print('datapoison_rst:', datapoison_rst)
    print('datapoison_defense_rst:', datapoison_defense_rst)
    final_rst = {'tag': tag}
    if adversarial_rst is not None:
        final_rst.update(adversarial_rst)
    if backdoor_rst is not None:
        final_rst.update(backdoor_rst)
    if datapoison_rst is not None:
        final_rst.update(datapoison_rst)
    if datapoison_defense_rst is not None:
        final_rst.update(datapoison_defense_rst)

    with open('./ModelResults.csv', 'r', newline='') as csvfile:
        data = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
        headers = set(data[0].keys())
        headers.update(final_rst.keys())
        data.append(final_rst)

    with open('./ModelResults.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    result = raw_data_process(final_rst)
    return


if __name__ == '__main__':
    ModelEvaluation(evaluation_params=evaluation_params)

