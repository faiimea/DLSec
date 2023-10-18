# FAdv

## Intro
对抗样本生成器  
目前实现攻击算法如下：
* FGSM: Fast Gradient Sign Method
* DIFGSM: Diverse-input Iterative FGSM
* MIFGSM: Momentum Iterative FGSM
* NIFGSM: Nesterov-accelerated Iterative FGSM
* SINIFGSM: Scale-invariant Nesterov-accelerated Iterative FGSM
* TIFGSM: Translation-invariant Iterative FGSM
* VMIFGSM: Variance-tuned Momentum Iterative FGSM
* VNIFGSM: Variance-tuned Nesterov Iterative FGSM
* PGD: Projected Gradient Descent
* DEEPFOOL - 似乎有一点点bug，但是能跑通

## Data
使用CIFAR-10进行测试(没有上传到Github)

## Test
```shell
python ./src/test.py
```
目前test.py默认使用FGSM算法
在调整算法时，需要手动修改
```python
attack = FGSM(model, eps=eps)
```
其中算法的名称。
可以调节`epsilons`参数来进行不同上限的测试。

TODO：
- [ ] parser参数传递器：选择模型+参数+迭代率
- [ ] DataLoader拓展+重构
- [ ] 拓展平台以满足Backdoor/Data Poison攻击与防御的需要
- [ ] 测试脚本补全
## 实验结果

在epsilon = [0,0.01,0.03,0.05,0.1] 的初始设置下，分别测试各个算法准确度

*Baseline采用cifar10_resnet56