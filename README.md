# DLSec

## 测试方法

修改**EvaluationConfig.py**内参数，主要需要修改的内容如下：

1. model：在此处加载模型
2. evaluation_params内的tag：当前模型的标签，**一定要修改**否则之后不知道结果对应的啥模型
3. evaluation_params内的use_dataset：可更改使用的数据集，目前暂时先用torchvision内支持的模型

修改好后运行**EvaluationPlatformNEW.py**，当前测试结果会合并到ModelResults.csv内



## Intro


### 对抗样本模块
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


