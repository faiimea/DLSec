import torch
import os
from Backdoor.Defense.base import BackdoorDefense
from Backdoor.Defense.Deepinspect import deepinspect
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision

import torchvision.datasets


def run_backdoor_defense(allow_defense=True, model=None, method="NeuralCleanse", train_dataloader=None, params=None):
    model.eval()
    if params['tag'] is None:
        this_turn_tag = method+"-"+ datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        this_turn_tag = params['tag']
    if method == 'NeuralCleanse' or method=='Tabor':
        DetectedBackdoor, trigger,ReinforcedModel, new_model_dict_path = BackdoorDefense(dataloader=train_dataloader, model=model, triggerpath="Backdoor/Defense/" +this_turn_tag+".pth",generator_path=params['generator_path'],load_generator=params['load_generator'])
    elif method == 'DeepInspect':
        DetectedBackdoor, trigger,ReinforcedModel, new_model_dict_path = deepinspect(model, train_dataloader, tag=this_turn_tag,generator_path=params['generator_path'],load_generator=params['load_generator'])
    else:
        print("请选择正确的后门检测方法:\n1:NeuralCleanse\t2:Tabor\t3:DeepInspect")
        return False, None, None
    backdoor_result={'backdoor_label':DetectedBackdoor,'trigger_std':trigger[0],'trigger_size':trigger[1]}
    if len(DetectedBackdoor) == 0:
        return False, backdoor_result, None
    else:
        return True, backdoor_result,new_model_dict_path
