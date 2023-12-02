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
"""

import torch
from Adv_Sample import *
from Backdoor.Defense import *
from Datapoison.Defense import *
import copy

evaluation_params = {
    'model_path': None
}


def ModelEvaluation(model_path: str = None):
    Model2BeEvaluated = torch.load(model_path)
    adversarial_rst = adversarial_test(Model2BeEvaluated)
    isBackdoored, backdoor_rst = backdoor_detect(Model2BeEvaluated)
    isPoisoned, datapoison_rst = datapoison_detect(Model2BeEvaluated)
    process_result(adversarial_rst, backdoor_rst, datapoison_rst)

    ReinforcedModel = copy.deepcopy(Model2BeEvaluated)
    if isBackdoored:
        backdoor_defense(ReinforcedModel)
    if isPoisoned:
        datapoison_defense(ReinforcedModel)
    reinforced_adversarial_rst = adversarial_test(Model2BeEvaluated)
    _, reinforced_backdoor_rst = backdoor_detect(ReinforcedModel)
    _, reinforced_datapoison_rst = datapoison_detect(ReinforcedModel)
    process_result(reinforced_adversarial_rst, reinforced_backdoor_rst, reinforced_datapoison_rst)


def adversarial_test(model):
    adversarial_rst = None
    return adversarial_rst


def backdoor_detect(model):
    isBackdoored = False
    backdoor_rst = None
    return isBackdoored, backdoor_rst


def datapoison_detect(model):
    isPoisoned = False
    datapoison_rst = None
    return isPoisoned, datapoison_rst


def backdoor_defense(model):
    return


def datapoison_defense(model):
    return


def process_result(adversarial_rst, backdoor_rst, datapoison_rst):
    return


if __name__ == "__main__":
    ModelEvaluation(**evaluation_params)
