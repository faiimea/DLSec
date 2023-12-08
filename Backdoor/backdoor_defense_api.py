import torch
import os
from Backdoor.Defense.base import BackdoorDefense

import torchvision.datasets

example_params = {
    'model_path': None,  # Model_to_be_Tested文件夹下相对路径
    'method': 'NeuralCleanse'
}


def run_backdoor_defense(model_path: str = None, method: str = None, use_dataset=torchvision.datasets.CIFAR10):
    model = torch.load("./Model_to_be_Tested/" + model_path)

    dataloader=None
    '''
    数据集 TO BE DONE
    '''
    if method=='NeuralCleanse':
        bdd = BackdoorDefense(dataloader=dataloader, model=model, triggerpath="./Backdoor/Defense/"+model_path)
        bdd.run()
        '''
        bdd返回的测试结果 TO BE DONE
        有待协商
        '''
    return


if __name__ == '__main__':
    run_backdoor_defense(**example_params)
