import torch
import torchvision
from tqdm import tqdm
from numpy import mean

import forest

'''
name_tag:用于命名文件夹
device:计算设备
local_model:应该传入一个预训练好的模型，如果策略为from-scratch就是我们先训练一下，然后再投毒，否则就是直接投毒
dataset:在什么数据集上投毒和训练，传入torchvision.datasets下数据集（这里自己封装了，只能填字符串，后续想想能不能改）
epochs:训练轮数
batch_size:训练模型时的批大小
poison_batch_size:投毒时的批大小
lr:训练模型的学习率
weight_decay:权重衰减
optimizer:优化器，使用torch.optim下的优化器
scenario:训练策略，可选['from-scratch', 完整训练
                    'transfer'] 从预训练模型迁移
poisonkey:投毒密钥，用于生成随机数
targets:投毒目标数量
budget:投毒比例
eps:允许修改的幅度
algorithm:投毒算法，可选['gradient-matching', 'gradient-matching-private', 'gradient-matching-mt',
          'watermark', 'poison-frogs', 'metapoison', 'hidden-trigger',
          'metapoison-v2', 'metapoison-v3', 'bullseye', 'patch',
          'gradient-matching-hidden', 'convex-polytope']（大概？）
restarts:投毒重启次数
attack_iter:投毒迭代次数
tau:投毒时的学习率
vruns:控制了在训练模型后，多少次重新初始化模型并检查是否达到了预定的目标（不太懂，先设默认的1）

生成毒药的优化器直接选了signAdam？
'''

path = 'E:\Projects\Pycharm\DLSec\data' # 不太确定工作路径设置的哪里，绝对路径反正没错

class DatapoisonAttack():
    def __init__(self, local_model=None, device: str = 'cuda:0', dataset: str = 'CIFAR10', epochs: int = 10,
                 batch_size: int = 128, poison_batch_size: int = 512, poisonkey: float = None, lr: float = 0.1,
                 weight_decay: float = 5e-4, optimizer: str = 'SGD', scenario: str = 'transfer', data_path: str = path,
                 tag: str = '', targets: int = 1, eps: float = 16.0, algorithm: str = 'poison-frogs', restarts: int = 3,
                 attack_iter: int = 200, tau: float = 0.1, vruns: int = 1):
        setup = {'device': torch.device(device), 'dtype': torch.float32, 'non_blocking': True}
        args = {'scenario': scenario, 'random_seed': None, 'local_model': local_model, 'optimizer': optimizer,
                'dataset': dataset, 'lr': lr, 'weight_decay': weight_decay, 'epochs': epochs, 'batch_size': batch_size,
                'poison_batch_size': poison_batch_size, 'poisonkey': poisonkey, 'data_path': data_path, 'tag': tag,
                'targets': targets, 'budget': 0.01, 'eps': eps, 'algorithm': algorithm, 'restarts': restarts,
                'attack_iter': attack_iter, 'tau': tau, 'vruns': vruns}

        if args['scenario'] == 'from-scratch':
            args['pretrained'] = False
        else:
            args['pretrained'] = True

        self.model = forest.Victim(args, setup=setup)
        self.data = forest.Kettle(args, args['batch_size'], augmentations=True, setup=setup)
        self.witch = forest.Witch(args, setup=setup)

        self.stats_clean = None
        if args['scenario'] == 'from-scratch':
            self.stats_clean = self.model.train(self.data, max_epoch=args['epochs'])

        # 生成毒药
        self.poison_delta = self.witch.brew(self.model, self.data)

        # 保存毒药
        # poison_delta = poison_delta.to(device='cpu')
        # self.data.export_poison(poison_delta, path='poisons/', mode='numpy')

    # 测试毒药
    def test(self, times: int = 1):
        poison_acc = []
        acc = []
        for _ in range(times):
            res = self.model.validate(self.data, self.poison_delta)   # 返回投毒成功率+投毒后的准确率？
            poison_acc = res['target_accs'][-1]
            acc = res['valid_accs'][-1]

        return poison_acc, acc

'''
if __name__ == "__main__":
    a = DatapoisonAttack(local_model=torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]),
                         scenario='from-scratch', epochs=2, attack_iter=20, restarts=1)
    print('开始测试：')
    results = a.test()
    print(results)
'''