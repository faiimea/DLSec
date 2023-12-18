import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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
