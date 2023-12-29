import torch
import os
from Backdoor.Defense.base import BackdoorDefense
from Backdoor.Defense.Deepinspect import deepinspect
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision

import torchvision.datasets


def run_backdoor_defense(allow_defense=True, model=None, method="DeepInspect", train_dataloader=None, params=None):
    if method == 'NeuralCleanse':
        bdd = BackdoorDefense(dataloader=train_dataloader, model=model, triggerpath="./Backdoor/Defense/" + method + datetime.now().strftime("-%Y%m%d-%H%M%S") + ".pth")
        bdd.run()
        '''
        bdd返回的测试结果 TO BE DONE
        有待协商
        '''
    elif method == 'DeepInspect':
        if params['tag'] is None:
            this_turn_tag = "DeepInspect-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            this_turn_tag = params['tag']
        DetectedBackdoor, ReinforcedModel, new_model_dict_path = deepinspect(model, train_dataloader, tag=this_turn_tag,generator_path=params['DEEPINSPECT_generator_path'],load_generator=params['DEEPINSPECT_load_generator'])

    if len(DetectedBackdoor) == 0:
        return False, None, None
    else:
        return True, DetectedBackdoor, new_model_dict_path
