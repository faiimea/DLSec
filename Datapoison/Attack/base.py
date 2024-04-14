import torch
import torchvision

from Datapoison.Attack import forest

'''
name_tag:用于命名文件夹
device:计算设备
model:应该传入一个预训练好的模型，如果策略为from-scratch就是我们先训练一下，然后再投毒，否则就是直接投毒
use_dataset:在什么数据集上投毒和训练，传入torchvision.datasets下数据集（这里自己封装了，只能填字符串，后续想想能不能改）
epochs:训练轮数
batch_size:训练模型时的批大小
poison_batch_size:投毒时的批大小
poison_lr:训练模型的学习率
weight_decay:权重衰减
optimizer:优化器，使用torch.optim下的优化器
scenario:训练策略，可选['from-scratch', 完整训练
                    'transfer'] 从预训练模型迁移
poisonkey:投毒密钥，用于生成随机数
poison_target_num:投毒目标数量
poison_budget:投毒比例
poison_eps:允许修改的幅度
datapoison_method:投毒算法，可选['gradient-matching', 'gradient-matching-private', 'gradient-matching-mt',
          'watermark', 'poison-frogs', 'metapoison', 'hidden-trigger',
          'metapoison-v2', 'metapoison-v3', 'bullseye', 'patch',
          'gradient-matching-hidden', 'convex-polytope']（大概？）
poison_restarts:投毒重启次数
poison_attack_iter:投毒迭代次数
poison_tau:投毒时的学习率
poison_vruns:控制了生成毒药后，验证毒药次数

生成毒药的优化器直接选了signAdam？
'''

path = r'.\\data'  # 不太确定工作路径设置的哪里，绝对路径反正没错



class DatapoisonAttack():
    def __init__(self, params):
        setup = {'device': params['device'], 'dtype': torch.float32, 'non_blocking': True}
        args = {
            'scenario': params['scenario'],
            'random_seed': None,
            'local_model': params['model'],
            'optimizer': params['poison_optimizer'],
            'dataset': params['use_dataset'],
            'lr': params['poison_lr'],
            'weight_decay': params['poison_weight_decay'],
            'epochs': params['poison_epoch'],
            'batch_size': params['batch_size'],
            'poison_batch_size': params['poison_batch_size'],
            'poisonkey': params['poison_key'],
            'data_path': path, 'tag': params['tag'],
            'targets': params['poison_target_num'],
            'budget': params['poison_budget'],
            'eps': params['poison_eps'],
            'algorithm': params['datapoison_method'],
            'restarts': params['poison_restarts'],
            'attack_iter': params['poison_attack_iter'],
            'tau': params['poison_tau'],
            'vruns': params['poison_vruns']
        }

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
            res = self.model.validate(self.data, self.poison_delta)   # 不确定测试时要训练几次，也许1次比较好，但是先按epochs来
            poison_acc = res['target_accs'][-1]
            acc = res['valid_accs'][-1]
        rst={"PoisonSR": poison_acc, "afterPoisonACC": acc}
        print("数据投毒结果",rst)
        return rst


'''
import sys
sys.path.append("../..")
from EvaluationConfig import evaluation_params

if __name__ == "__main__":
    a = DatapoisonAttack(evaluation_params)
    print('开始测试：')
    results = a.test()
    print(results)
'''