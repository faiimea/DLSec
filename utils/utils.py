import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np


def evaluate(data_loader, model, criterion=torch.nn.CrossEntropyLoss(), device=torch.device('cuda')):
    model.to(device)
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


def cal_entropy_weight(data: pd.DataFrame):
    for column in data.columns:
        sum_p = sum(data[column])
        data[column] = data[column].apply(lambda x: x / sum_p if x / sum_p != 0 else 1e-9)
    E = (-1 / np.log(data.index.size)) * np.array([sum([p_ij * np.log(p_ij) for p_ij in data[column_j]]) for column_j in data.columns])
    E = pd.Series(E, index=data.columns)
    d = pd.Series(1 - E, index=data.columns)
    w = d / sum(d)
    return w


if __name__ == "__main__":
    test_data = [[12, 44, 59], [19, 34, 28], [15, 45, 55], [21, 22, 33]]
    df = pd.DataFrame(test_data, columns=['c1', 'c2', 'c3'])
    weight = cal_entropy_weight(df)
    print(weight)
