import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pickle
import torch.nn.functional as F
import os

class NeuralCleanse():
    def __init__(self, X, Y, model, num_samples,path='/default'):
        self.X = X
        self.Y = Y
        self.num_classes = 10
        self.model = model
        self.triggers = []
        self.num_samples_per_label = num_samples
        self.possible_target_label=[]
        self.path=path
        if not os.path.exists('.'+path):
            os.makedirs('.'+path)
        if not os.path.exists('.' + path+'/triggers'):
            os.makedirs('.' +path+'/triggers')


    def draw_trigger(self, M, Delta, file_name):
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

    def plot_metrics(self, loss_dict, file_name):
        plt.cla()
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        for i, metric in enumerate(loss_dict.keys()):
            y = loss_dict[metric]
            x = range(len(y))
            ax.plot(x, y, label=metric)
        ax.set_xlabel('epoch')
        ax.legend(loss_dict.keys(), loc='upper left')
        plt.tight_layout()
        plt.savefig(file_name + ".png")
        plt.close()

    def reverse_engineer_triggers(self):
        for target_label in range(self.num_classes):
            print(" ----------------------    reverse engineering the possible trigger for label ", target_label,
                  "   -----------------------")
            x_samples = []
            for base_label in range(self.num_classes):
                if base_label == target_label:
                    continue
                possible_idx = (np.where(self.Y == base_label)[0]).tolist()
                idx = random.sample(possible_idx, min(self.num_samples_per_label, int(len(possible_idx)/4)))
                x_samples.append(self.X[idx,::])


            x_samples = np.vstack(x_samples)
            y_t = np.ones((x_samples.shape[0])) * target_label
            y_t = torch.nn.functional.one_hot(torch.tensor(y_t, dtype=torch.long), self.num_classes).float().to("cuda")
            opt_round = 250
            m = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2],1)), dtype=torch.float32,requires_grad=True)
            delta = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2], self.X.shape[3])),dtype=torch.float32,requires_grad=True)
            lmbda=0.03
            delta_opt=torch.optim.Adam([delta],lr=0.5)
            m_opt = torch.optim.Adam([m], lr=0.5)
            no_improvement = 0
            patience = 10
            best_loss = 1e+10
            loss_dict = {'loss': []}

            # Define the training loop
            for r in range(opt_round):
                delta_opt.zero_grad()
                m_opt.zero_grad()
                poisoned_x=torch.tensor(x_samples,dtype=torch.float32)
                poisoned_x=poisoned_x*(1-m)+m*delta
                # poisoned_x=poisoned_x.to("cuda")
                poisoned_x = poisoned_x.permute(0, 3, 1, 2).to("cuda")
                prediction = self.model(poisoned_x)
                loss = F.cross_entropy(prediction, y_t) + (lmbda * torch.sum(torch.abs(m))).to("cuda")
                if loss<best_loss:
                    no_improvement = 0
                    best_loss = loss
                else:
                    no_improvement += 1

                if no_improvement == patience:
                    print("\nDecreasing learning rates...")
                    for g in delta_opt.param_groups:
                        g['lr'] /= 10.0
                    for g in m_opt.param_groups:
                        g['lr'] /= 10.0

                loss_dict['loss'].append(loss.item())
                print("[", str(r), "] loss:", "{0:.5f}".format(loss), end='\r')

                loss.backward()
                delta_opt.step()
                m_opt.step()
                with torch.no_grad():
                    # 防止trigger和norm越界
                    torch.clip_(m, 0, 1)
                    torch.clip_(delta, 0, 1)
                lmbda = 0.03*(r-no_improvement)/(opt_round-no_improvement)+0.03
                if r % 50 == 0:
                    self.plot_metrics(loss_dict=loss_dict, file_name='.'+self.path+'/opt-history')
                    self.draw_trigger(M=m.detach().clone(), Delta=m.detach().clone(),
                                      file_name='.'+self.path+'/triggers/trigger-' + str(target_label))
                    with torch.no_grad():

                        poisoned_x = torch.tensor(x_samples, dtype=torch.float32)
                        # poisoned_x=(poisoned_x*(1-m)+m*delta).to("cuda")
                        poisoned_x = (poisoned_x * (1 - m) + m * delta).to("cuda").permute(0, 3, 1, 2)
                        bckdr_acc = (torch.argmax(self.model(poisoned_x), dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()

                    print("\nbackdoor accuracy:", "{0:.2f}".format(bckdr_acc))
            trigger = (delta.detach().clone(), m.detach().clone(),torch.sum(torch.abs(m)).item())
            self.triggers.append(trigger)
        with open('.'+self.path+'/triggers.npy', 'wb') as f:
            pickle.dump(self.triggers,f)



    def backdoor_detection(self):

        if len(self.triggers) != self.num_classes:
            try:
                with open('.'+self.path+'/triggers.npy','rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()

        _, _, l1_norms = zip(*self.triggers)
        stringency = 1.5
        median = np.median(l1_norms, axis=0)
        MAD = 1.4826 * np.median(np.abs(l1_norms - median), axis=0)
        in_ditribution = (median - stringency * MAD < l1_norms) * (l1_norms < median + stringency * MAD)
        out_of_distribution = np.logical_not(in_ditribution)
        outliers = np.where(out_of_distribution)[0]

        for possible_target_label in outliers:
            x_samples = []
            for base_label in range(self.num_classes):
                if base_label == possible_target_label:
                    continue
                possible_idx = (np.where(self.Y == base_label)[0]).tolist()
                idx = random.sample(possible_idx, min(self.num_samples_per_label, len(possible_idx)))
                x_samples.append(self.X[idx, ::])

            x_samples = np.vstack(x_samples)
            delta=self.triggers[possible_target_label][0]
            m=self.triggers[possible_target_label][1]
            poisoned_x = torch.tensor(x_samples, dtype=torch.float32)
            poisoned_x = poisoned_x * (1 - m) + m * delta
            # poisoned_x = poisoned_x.to("cuda")
            poisoned_x = poisoned_x.permute(0, 3, 1, 2).to("cuda")
            y_t = (np.ones((poisoned_x.shape[0])) * possible_target_label)
            # y_t = keras.utils.to_categorical(y_t, self.num_classes)
            y_t = torch.nn.functional.one_hot(torch.tensor(y_t, dtype=torch.long), self.num_classes).float().to("cuda")
            bckdr_acc = (torch.argmax(self.model(poisoned_x), dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()
            if 0.75 < bckdr_acc:
                self.possible_target_label.append(possible_target_label)
                print("There is a possible backdoor to label ", possible_target_label, " with ",
                  "{0:.2f}".format(100 * bckdr_acc), "% accuracy.")
    def mitigate(self,test_X,test_Y):
        BATCH_SIZE=64
        TARGET_LS = self.possible_target_label
        NUM_LABEL = len(TARGET_LS)
        PER_LABEL_RARIO = 0.1
        INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
        def injection_func(mask, pattern, adv_img):
            return mask * pattern + (1 - mask) * adv_img
        def infect_X(img, tgt):
            try:
                with open('.'+self.path+'/triggers.npy', 'rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()
            mask, pattern = self.triggers[4][1].numpy(),self.triggers[4][0].numpy()
            raw_img = np.copy(img)
            adv_img = np.copy(raw_img)
            adv_img = injection_func(mask, pattern, adv_img)
            return adv_img, tgt

        class DataGenerator():
            def __init__(self, target_ls, X, Y, inject_ratio,
                         is_test=0):  # target_ls is list of all possible targets (constrained to length 1 in this implementation)
                self.target_ls = target_ls
                self.X = X
                self.Y = Y
                self.inject_ratio = inject_ratio
                self.is_test = is_test
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
                        tgt = random.choice(self.target_ls)
                        cur_x, _ = infect_X(cur_x, tgt)
                    batch_X.append(cur_x)
                    batch_Y.append(cur_y)
                    if len(batch_Y) == BATCH_SIZE:
                        batch_X = torch.from_numpy(np.array(batch_X))
                        batch_Y = torch.from_numpy(np.array(batch_Y))
                        return batch_X.float(), batch_Y.long()
                    elif self.idx == len(self.Y):
                        return (torch.from_numpy(np.array(batch_X)).float(), torch.from_numpy(np.array(batch_Y)).long())

        train_size= len(self.Y)
        train_X = self.X
        train_Y = self.Y
        train_gen = DataGenerator(TARGET_LS, train_X, train_Y, INJECT_RATIO, 0)
        test_adv_gen = DataGenerator(TARGET_LS, test_X, test_Y, 1, 1)
        test_clean_gen = DataGenerator(TARGET_LS, test_X, test_Y, 0, 1)
        loss = torch.nn.CrossEntropyLoss()
        lr = 0.0005
        def fit(model,train_gen, verbose, steps_per_epoch, learning_rate,loss, change_lr_every=25,test_gen = None, stps = None, model_path = None):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
            epoch=0
            Accuracy=0
            while Accuracy<0.995:
                if (epoch % change_lr_every == change_lr_every - 1):
                    learning_rate = learning_rate / 2
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                train_gen.on_epoch()
                running_loss = 0.0
                y_pred = []
                y_act = []
                for step in range(steps_per_epoch):
                    data_x, data_y = train_gen.gen_data()
                    optimizer.zero_grad()
                    data_x = data_x.to("cuda")
                    data_y = data_y.to("cuda")
                    data_x = data_x.permute(0, 3, 1, 2)
                    out = model.forward(data_x)
                    lossF = loss(out, data_y)
                    lossF.backward()
                    optimizer.step()
                    running_loss += lossF.item()
                    y_pred.append(torch.argmax(out, dim=1).cpu().numpy())
                    y_act.append(data_y.cpu().numpy())
                y_pred = np.array(y_pred).flatten()
                y_act = np.array(y_act).flatten()

                Accuracy = (sum([y_pred[i] == y_act[i] for i in range(len(y_pred))])) / len(y_pred)
                epoch+=1
                if (verbose):
                    # print(running_loss, steps_per_epoch)
                    print("Epoch -- {} ; Average Loss -- {} ; Accuracy -- {}".format(epoch,
                                                                                     running_loss / (steps_per_epoch),
                                                                                     Accuracy))
            print("Training Done")
            return
        def evaluate(model, test_gen, steps_per_epoch, loss, verbose, mask = None, pattern = None):
            running_loss = 0.0
            y_pred = []
            y_act = []

            test_gen.on_epoch()
            model.eval()

            for step in range(steps_per_epoch):
                with torch.no_grad():
                    if mask is not None:
                        data_x, data_y = test_gen.gen_data(mask=mask, pattern=pattern)
                    else:
                        data_x, data_y = test_gen.gen_data()
                    data_x = data_x.to("cuda")
                    data_y = data_y.to("cuda")
                    data_x = data_x.permute(0, 3, 1, 2)
                    out = model.forward(data_x)

                    y_pred.append(torch.argmax(out, dim=1).cpu().numpy())
                    y_act.append(data_y.cpu().numpy())

                    lossF = loss(out, data_y)

                    running_loss += lossF.item()

            y_pred = np.array(y_pred).flatten()
            y_act = np.array(y_act).flatten()

            Accuracy = (sum([y_pred[i] == y_act[i] for i in range(len(y_pred))])) / len(y_pred)
            running_loss /= steps_per_epoch

            if (verbose):
                print("Accuracy on provided Data -- {} ; Loss -- {}".format(Accuracy, running_loss))

            return Accuracy, running_loss
        fit(self.model,train_gen, verbose=1, steps_per_epoch=int(train_size // BATCH_SIZE), learning_rate=lr,loss=loss, change_lr_every=35)
        print("Evaluating model")
        number_images = len(test_Y)
        steps_per_epoch = int(number_images // BATCH_SIZE)
        acc, _ =evaluate(self.model,test_clean_gen, steps_per_epoch, loss, 1)
        backdoor_acc, _ =evaluate(self.model,test_adv_gen, steps_per_epoch, loss, 1)
        print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


