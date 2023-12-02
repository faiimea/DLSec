'''
这里其实不用写太多，因为主要的训练和测试函数都在Attack.base中完成
主要写参数传递和调用防御就好
'''
import numpy as np
from .NeuralCleanse import NeuralCleanse
import os


class BackdoorDefense():
    def __init__(self,dataloader,model,triggerpath='default'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.model=model
        self.path='/'+triggerpath
        org_img = []
        org_label = []
        for data in dataloader:
            image, label = data
            org_img.append(image[0].permute(1, 2, 0).numpy())
            org_label.append(int(label[0]))
        org_img = np.array(org_img)
        org_label = np.array(org_label)
        self.X=org_img
        self.Y=org_label
        self.NC = NeuralCleanse(X=org_img, Y=org_label, model=model, num_samples=25, path=self.path)
    def run(self,alreadyreverse=False):
        if not alreadyreverse:
            self.NC.reverse_engineer_triggers()
        self.NC.draw_all_triggers()
        self.NC.backdoor_detection()


if __name__ == "__main__":
    pass
