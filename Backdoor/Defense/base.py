'''
这里其实不用写太多，因为主要的训练和测试函数都在Attack.base中完成
主要写参数传递和调用防御就好
'''
import numpy as np
from .NeuralCleanse import NeuralCleanse
import os
import torch

def BackdoorDefense(dataloader,model,method='NC',triggerpath='default'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    path='/'+triggerpath
    org_img = []
    org_label = []
    for data in dataloader:
        image, label = data
        org_img.append(image[0].permute(1, 2, 0).numpy())
        org_label.append(int(label[0]))
    org_img = np.array(org_img)
    org_label = np.array(org_label)
    num_classes = len(set(org_label))
    if method=='NeuralCleanse':
        DF = NeuralCleanse(X=org_img, Y=org_label, model=model, num_samples=10, num_classes=num_classes,path=path)
    elif method=='Tabor':
        DF = NeuralCleanse(X=org_img, Y=org_label, model=model, num_samples=10, num_classes=num_classes,path=path)
    DF.reverse_engineer_triggers()
    outlier,trigger=DF.backdoor_detection()
    DF.mitigate(test_X=org_img,test_Y=org_label)
    new_model_save_path = os.getcwd() + "/"+triggerpath
    torch.save(DF.model.state_dict(), new_model_save_path)
    return outlier,trigger,DF.model,new_model_save_path
    



if __name__ == "__main__":
    pass
