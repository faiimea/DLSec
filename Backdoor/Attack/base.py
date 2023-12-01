import torch
from datetime import datetime
from typing import Any
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

'''
tag:保存模型时命名备注
device:计算设备
model:被攻击模型
clean_dataset:在什么数据集上投毒和训练，传入torchvision.datasets下数据集，或一个字符串路径表示使用自己的数据集
poison_rate:投毒率
lr:学习率
target_label:目标标签
epochs:训练轮数
batch_size:批大小
optimizer:优化器，使用torch.optim下的优化器
local_model_path:默认为None，或传入字符串路径加载已有模型，此时model失效
'''
project_data_path = "../data"


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("Not supported, using Adam")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def build_transform(mode):
    if len(mode) == 3 and mode[2] == 3:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        channels = 3
    elif len(mode) == 2:
        mean, std = (0.5,), (0.5,)
        channels = 1
    elif len(mode) == 3 and mode[2] == 4:
        mean, std = (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
        channels = 4
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    return transform, detransform, channels


def train_one_epoch(data_loader, model, optimizer, criterion, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {"loss": running_loss / len(data_loader)}


def evaluate(data_loader, model, criterion, device):
    model.eval()
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    loss = sum(loss_sum) / len(loss_sum)
    return {"acc": accuracy_score(y_true.cpu(), y_predict.cpu()), "loss": loss}


def evaluate_model(clean_test_data, poisoned_test_data, model, criterion, device):
    clean_stat = evaluate(clean_test_data, model, criterion, device)
    poisoned_stat = evaluate(poisoned_test_data, model, criterion, device)
    return {'CDA': clean_stat['acc'], 'clean_data_loss': clean_stat['loss'], 'ASR': poisoned_stat['acc'], 'poisoned_data_loss': poisoned_stat['loss']}


class BackdoorAttack:
    def __init__(self, tag: str = 'CustomModel', device: str = 'cpu', model=None, clean_dataset=None, poison_rate: float = 0.05, lr: float = 0.1, target_label=2, epochs: int = 20, batch_size: int = 64, optimizer: str = None, criterion=torch.nn.CrossEntropyLoss(),
                 local_model_path: str = None):
        self.data_path = project_data_path
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S-") + tag + ".pth"
        if isinstance(clean_dataset, str):
            print("自定义数据集仍在开发中")
            raise NotImplementedError
        else:
            self.clean_traindata = clean_dataset(self.data_path, train=True, download=True)
            self.clean_testdata = clean_dataset(self.data_path, train=False, download=True)
        self.transform, self.detransform, self.channels = build_transform(self.clean_traindata.data.shape[1:])
        self.data_shape=self.clean_traindata.data.shape[1:]
        self.clean_traindata.transform = self.transform
        self.clean_testdata.transform = self.transform

        if local_model_path is not None:
            print("Loading local model {",local_model_path,"}")
            self.model = torch.load("./LocalModels/"+local_model_path).to(self.device)
            self.model_path = "checkpoints/" + local_model_path
        else:
            self.model = model.to(self.device)
        self.poison_rate = poison_rate
        self.lr = lr
        self.target_label = target_label
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer_picker(optimizer, self.model.parameters(), lr=self.lr)
        self.criterion = criterion
        self.train_one_epoch = train_one_epoch
        self.evaluate_model = evaluate_model

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model.forward(*args, **kwds)


if __name__ == "__main__":
    pass
