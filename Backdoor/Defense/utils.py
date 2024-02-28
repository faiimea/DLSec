import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

def evaluate(model, test_gen, steps_per_epoch, loss, verbose, device='cpu', mask=None, pattern=None):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    test_gen.on_epoch()
    model.eval()
    for step in range(steps_per_epoch):
        with torch.no_grad():
            if mask is not None:
                data_x, data_y = test_gen.gen_data(mask=mask, pattern=pattern)
            else:
                data_x, data_y = test_gen.gen_data()
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            data_x = data_x.permute(0, 3, 1, 2)
            out = model.forward(data_x)
            predictions = torch.argmax(out, dim=1)
            # 计算正确的预测数量
            correct_predictions += (predictions == data_y).sum().item()
            total_predictions += data_y.size(0)
            lossF = loss(out, data_y)
            running_loss += lossF.item()
    running_loss /= steps_per_epoch
    Accuracy = correct_predictions / total_predictions
    if (verbose):
        print("Accuracy on provided Data -- {} ; Loss -- {}".format(Accuracy, running_loss))
    return Accuracy, running_loss


def fit(model, train_gen, verbose, steps_per_epoch, learning_rate, loss, device='cpu', change_lr_every=25):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    for epoch in range(20):
        if (epoch % change_lr_every == change_lr_every - 1):
            learning_rate = learning_rate / 5
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        train_gen.on_epoch()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        pbar = tqdm(range(steps_per_epoch))
        for _ in pbar:
            data_x, data_y = train_gen.gen_data()
            optimizer.zero_grad()
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            data_x = data_x.permute(0, 3, 1, 2)
            out = model.forward(data_x)
            lossF = loss(out, data_y)
            lossF.backward()
            optimizer.step()
            running_loss += lossF.item()
            predictions = torch.argmax(out, dim=1)
            # 计算正确的预测数量
            correct_predictions += (predictions == data_y).sum().item()
            total_predictions += data_y.size(0)

        Accuracy = correct_predictions / total_predictions
        if (verbose):
            # print(running_loss, steps_per_epoch)
            print("Epoch -- {} ; Average Loss -- {} ; Accuracy -- {}".format(epoch,
                                                                             running_loss / (steps_per_epoch),
                                                                             Accuracy))
    print("Training Done")
    return


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


class DataGenerator():
    def __init__(self, mask, pattern, target_ls, X, Y, inject_ratio, BATCH_SIZE,
                 is_test=0):  # target_ls is list of all possible targets (constrained to length 1 in this implementation)
        self.mask = mask
        self.pattern = pattern
        self.target_ls = target_ls
        self.X = X
        self.Y = Y
        self.inject_ratio = inject_ratio
        self.is_test = is_test
        self.BATCH_SIZE = BATCH_SIZE
        self.idx = 0
        self.indexes = np.arange(len(self.Y))

    def on_epoch(self):
        self.idx = 0
        if (self.is_test == 0):
            np.random.shuffle(self.indexes)

    def gen_data(self):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = self.indexes[self.idx]
            self.idx += 1
            cur_x = self.X[cur_idx]
            cur_y = self.Y[cur_idx]
            if inject_ptr < self.inject_ratio:
                cur_x = infect_X(cur_x, self.mask, self.pattern)
            batch_X.append(cur_x)
            batch_Y.append(cur_y)
            if len(batch_Y) == self.BATCH_SIZE:
                batch_X = torch.from_numpy(np.array(batch_X))
                batch_Y = torch.from_numpy(np.array(batch_Y))
                return batch_X.float(), batch_Y.long()
            elif self.idx == len(self.Y):
                return (torch.from_numpy(np.array(batch_X)).float(), torch.from_numpy(np.array(batch_Y)).long())


def infect_X(img, mask, pattern):
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)
    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img

def draw_trigger( M, Delta, file_name):
    plt.cla()
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.imshow(Delta)
    ax.set_title('Delta')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 3, 2)
    ax.imshow(M.reshape(M.shape[0], M.shape[1]), cmap='gray')
    ax.set_title('M')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 3, 3)
    ax.imshow(M * Delta)
    ax.set_title('M*Delta')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.close()
