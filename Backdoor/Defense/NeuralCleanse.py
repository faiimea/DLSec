import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import cv2
import pickle
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import os
def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def normalize_image(image):
    normalized_image = image / 255.0
    return normalized_image

class NeuralCleanse():
    def __init__(self, X, Y, model, num_samples,path='default'):
        self.X = X
        self.Y = Y
        self.num_classes = np.max(Y)+1
        self.model = model
        self.triggers = []
        self.X_min = np.min(self.X)
        self.X_max = np.max(self.X)
        self.num_samples_per_label = num_samples
        self.path=path
        if not os.path.exists('./Defense'+path):
            os.makedirs('./Defense'+path)
        if not os.path.exists('./Defense' + path+'/triggers'):
            os.makedirs('./Defense' +path+'/triggers')

    def trigger_insert(self, X, Delta, M):
        return torch.clamp((1.0 - M) * X + M * Delta, self.X_min, self.X_max)

    def draw_trigger(self, M, Delta, file_name):
        plt.cla()
        plt.figure()
        # Delta=np.transpose(Delta, (1, 2, 0))
        # M=np.transpose(M, (1, 2, 0))
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
            opt_round = 500
            m = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2],1)), dtype=torch.float32,requires_grad=True)
            delta = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2], self.X.shape[3])),dtype=torch.float32,requires_grad=True)
            lmbda = torch.tensor(0.03,requires_grad=True)
            m_opt = torch.optim.Adam([m,lmbda], lr=0.5)
            delta_opt=torch.optim.Adam([delta],lr=0.5)

            no_improvement = 0
            patience = 10
            best_loss = 1e+10
            loss_dict = {'loss': []}

            # Define the training loop
            for r in range(opt_round):
                m_opt.zero_grad()
                delta_opt.zero_grad()
                # poisoned_x = self.trigger_insert(X=x_samples, Delta=delta.detach().clone(), M=m.detach().clone()).to("cuda")
                poisoned_x=[]
                for x in x_samples:
                    poisoned_x.append((1-m.detach().numpy())*x+m.detach().numpy()*delta.detach().numpy())
                poisoned_x=torch.tensor(np.array(poisoned_x),dtype=torch.float32).to("cuda").permute(0,3,1,2)
                prediction = self.model(poisoned_x)
                # bckdr_acc = (torch.argmax(prediction, dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()
                # print(bckdr_acc)
                loss = F.cross_entropy(prediction, y_t) + (lmbda * torch.sum(torch.abs(torch.sigmoid(m)))).to("cuda")
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
                # print("[", str(r), "] loss:", "{0:.5f}".format(loss))

                loss.backward()
                delta_opt.step()
                m_opt.step()
                if r % 50 == 0:
                    self.plot_metrics(loss_dict=loss_dict, file_name='./Defense'+self.path+'/opt-history')
                    self.draw_trigger(M=m.detach().clone(), Delta=delta.detach().clone(),
                                      file_name='./Defense'+self.path+'/triggers/trigger-' + str(target_label))
                    with torch.no_grad():
                        # poisoned_x = self.trigger_insert(X=x_samples, Delta=torch.sigmoid(delta.detach().clone()), M=torch.sigmoid(m.detach().clone())).to("cuda")
                        poisoned_x = []
                        for x in x_samples:
                            poisoned_x.append(
                                (1 - m.detach().numpy()) * x + m.detach().numpy() * delta.detach().numpy())
                        poisoned_x = torch.tensor(np.array(poisoned_x), dtype=torch.float32).to("cuda").permute(0,3,1,2)
                        bckdr_acc = (torch.argmax(self.model(poisoned_x), dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()

                    print("\nbackdoor accuracy:", "{0:.2f}".format(bckdr_acc))
            # trigger = (torch.sigmoid(delta).item(), torch.sigmoid(m).item(), torch.sum(torch.abs(torch.sigmoid(m))).item())
            # self.triggers.append(trigger)
            self.triggers.append((K.get_value(delta.detach().clone()), K.get_value(m.detach().clone()), K.get_value(K.sum(K.abs(K.sigmoid(m.detach().clone()))))))
        with open('./Defense'+self.path+'/triggers.npy', 'wb') as f:
            pickle.dump(self.triggers,f)

    def draw_all_triggers(self):

        if len(self.triggers) != self.num_classes:
            try:
                with open('./Defense'+self.path+'/triggers.npy', 'rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()

        plt.cla()
        plt.figure(figsize=(43, 43))
        for i in range(self.num_classes):
            ax = plt.subplot(4, 4, i + 1)
            ax.imshow(self.triggers[i][0] * self.triggers[i][1])
            ax.set_title("Class " + str(i) + " L1:" + "{0:.2f}".format(self.triggers[i][2]), fontsize=36)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig('./Defense'+self.path+"/all_triggers.png")
        plt.close()

    def backdoor_detection(self):

        if len(self.triggers) != self.num_classes:
            try:
                with open('./Defense'+self.path+'/triggers.npy','rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()

        _, _, l1_norms = zip(*self.triggers)

        stringency = 3.0
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
            poisoned_x_samples = []
            for x in x_samples:
                poisoned_x_samples.append(
                    (1 - m.detach().numpy()) * x + m.detach().numpy() * delta.detach().numpy())
            poisoned_x_samples = torch.tensor(np.array(poisoned_x_samples), dtype=torch.float32).to("cuda").permute(0,3,1,2)
            y_t = (np.ones((poisoned_x_samples.shape[0])) * possible_target_label)
            # y_t = keras.utils.to_categorical(y_t, self.num_classes)
            y_t = torch.nn.functional.one_hot(torch.tensor(y_t, dtype=torch.long), self.num_classes).float().to("cuda")
            bckdr_acc = (torch.argmax(self.model(poisoned_x_samples), dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()
            if 0.75 < bckdr_acc:
                print("There is a possible backdoor to label ", possible_target_label, " with ",
                      "{0:.2f}".format(100 * bckdr_acc), "% accuracy.")