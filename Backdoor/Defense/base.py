'''
这里其实不用写太多，因为主要的训练和测试函数都在Attack.base中完成
主要写参数传递和调用防御就好
'''
import numpy as np
from .NeuralCleanse import NeuralCleanse
from .Tabor import Tabor
import shutil
import os
import torch


def BackdoorDefense(dataloader, model, method='NeuralCleanse', triggerpath='default', generator_path=None, load_generator=False):
    path = '/' + triggerpath
    org_img = []
    org_label = []
    for data in dataloader.dataset:
        image, label = data
        org_img.append(image.permute(1, 2, 0).numpy())
        org_label.append(int(label))
    org_img = np.array(org_img)
    org_label = np.array(org_label)
    num_classes = len(set(org_label))
    DF = None
    if method == 'NeuralCleanse':
        DF = NeuralCleanse(X=org_img, Y=org_label, model=model, num_samples=25, num_classes=num_classes, path=path)
    elif method == 'Tabor':
        DF = Tabor(X=org_img, Y=org_label, model=model, num_samples=25, num_classes=num_classes, path=path)
    else:
        print("请选择正确的后门检测方法:\n1:NeuralCleanse\t2:Tabor\t3:DeepInspect")
    if not load_generator:
        DF.reverse_engineer_triggers()
    else:
        shutil.copy(generator_path, '.' + path + "/triggers.npy")

    outlier, trigger = DF.backdoor_detection()
    if outlier is None:
        return [], [0,0], None, None

    newmodel=DF.mitigate()
    new_model_save_path = os.getcwd() + "/" + triggerpath+'/defense_rst.pth'
    torch.save(newmodel.state_dict(), new_model_save_path)
    return outlier, trigger, newmodel, new_model_save_path


if __name__ == "__main__":
    pass
