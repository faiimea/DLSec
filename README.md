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
* DEEPFOOL - 暂时没有通过测试，准备之后找找bug

## Data
使用CIFAR-10进行测试(没有上传到Github)

## 实验结果

在epsilon = [0,0.01,0.03,0.05,0.1] 的初始设置下，分别测试各个算法准确度

*Baseline采用cifar10_resnet56
