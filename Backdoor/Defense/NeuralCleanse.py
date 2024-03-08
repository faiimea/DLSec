import pickle
import torch.nn.functional as F
import scipy.stats as stats
from .utils import *
import os
import torch
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

class NeuralCleanse():
    def __init__(self, X, Y, model,num_samples, num_classes=10,path='/default'):
        self.X = X
        self.Y = Y
        self.num_classes = num_classes
        self.model = model
        self.triggers = []
        self.num_samples_per_label = num_samples
        self.possible_target_label=[]
        self.path=path
        self.device=next(self.model.parameters()).device
        if not os.path.exists('.'+path):
            os.makedirs('.'+path)
        if not os.path.exists('.' + path+'/triggers'):
            os.makedirs('.' +path+'/triggers')
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
            y_t = torch.nn.functional.one_hot(torch.tensor(y_t, dtype=torch.long), self.num_classes).float().to(self.device)
            opt_round = 100
            m = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2],1)), dtype=torch.float32,requires_grad=True)
            delta = torch.tensor(np.random.uniform(0.0, 1.0, (self.X.shape[1], self.X.shape[2], self.X.shape[3])),dtype=torch.float32,requires_grad=True)
            lmbda=0.03
            delta_opt=torch.optim.Adam([delta],lr=0.5)
            m_opt = torch.optim.Adam([m], lr=0.5)
            no_improvement = 0
            patience = 10
            best_loss = 1e+10
            # Define the training loop
            pbar = tqdm(range(opt_round))

            for r in pbar:
                delta_opt.zero_grad()
                m_opt.zero_grad()
                x=torch.tensor(x_samples,dtype=torch.float32)
                poisoned_x=x*(1-m)+m*delta
                poisoned_x = poisoned_x.permute(0, 3, 1, 2).to(self.device)
                prediction = self.model(poisoned_x)
                loss = F.cross_entropy(prediction, y_t) + (lmbda * torch.sum(torch.abs(m))).to(self.device)
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

                pbar.set_postfix({'loss': loss})
                loss.backward()
                delta_opt.step()
                m_opt.step()
                with torch.no_grad():
                    # 防止trigger和norm越界
                    torch.clip_(m, 0, 1)
                    torch.clip_(delta, 0, 1)
                lmbda = 0.03*(r+no_improvement)/(opt_round+no_improvement)+0.03
            draw_trigger(M=m.detach().clone(), Delta=delta.detach().clone(),
                              file_name='.'+self.path+'/triggers/trigger-' + str(target_label))
            with torch.no_grad():
                poisoned_x = (x * (1 - m) + m * delta).to(self.device).permute(0, 3, 1, 2)
                bckdr_acc = (torch.argmax(self.model(poisoned_x), dim=1) == torch.argmax(y_t,dim=1)).float().mean().item()
            print("\nbackdoor accuracy:", "{0:.2f}".format(bckdr_acc))
            trigger = (delta.detach().clone(), m.detach().clone(),torch.sum(torch.abs(m)).item(),bckdr_acc)
            self.triggers.append(trigger)
        with open('.'+self.path+'/triggers.npy', 'wb') as f:
            pickle.dump(self.triggers,f)



    def backdoor_detection(self):
        if len(self.triggers) != self.num_classes:
            try:
                print('.' + self.path + '/triggers.npy')
                with open('.'+self.path+'/triggers.npy','rb') as f:
                    self.triggers = pickle.load(f)
            except Exception as e:
                print(e)
                print("you need to reverse engineer triggers, first.")
                exit()
        delta, _, l1_norms,acc = zip(*self.triggers)
        stringency = 1.5
        median = np.median(l1_norms, axis=0)
        MAD = 1.4826 * np.median(np.abs(l1_norms - median), axis=0)
        out_of_distribution=(median - stringency * MAD > l1_norms) 
        outliers = np.where(out_of_distribution)[0]
        result= stats.norm.cdf((median-l1_norms)/MAD)

        for possible_target_label in outliers:
            if 0.75 < acc[possible_target_label]:
                self.possible_target_label.append(possible_target_label)
                print("There is a possible backdoor to label ", possible_target_label, " with ",
                  "{0:.2f}".format(100 * acc[possible_target_label]), "% accuracy.")
        relative_size=delta[0].shape[2]*25
        if len(outliers>0):
            return outliers,[np.max(result),relative_size/np.min(l1_norms)]
        return None,[np.max(result),relative_size/np.min(l1_norms)]
    def mitigate(self):
        BATCH_SIZE = 64
        TARGET_LS = self.possible_target_label

        NUM_LABEL = len(TARGET_LS)
        if NUM_LABEL==0:
            print("there is no backdoor label in this model")
            return
        PER_LABEL_RARIO = 0.2
        INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
        train_X, train_Y, test_X, test_Y = train_test_split(
            self.X, self.Y,
            train_size=0.8, shuffle=True,
            stratify=self.Y
        )
        train_size = len(train_Y)
        mask, pattern = self.triggers[TARGET_LS[0]][1].numpy(), self.triggers[TARGET_LS[0]][0].numpy()
        train_gen = DataGenerator(mask, pattern,TARGET_LS, train_X, train_Y, inject_ratio=INJECT_RATIO, BATCH_SIZE=BATCH_SIZE,is_test=0)
        test_adv_gen = DataGenerator(mask, pattern,TARGET_LS, test_X, test_Y,inject_ratio=1, BATCH_SIZE=BATCH_SIZE, is_test=1)
        test_clean_gen = DataGenerator(mask, pattern,TARGET_LS, test_X, test_Y, inject_ratio=0,BATCH_SIZE=BATCH_SIZE, is_test=1)
        loss = torch.nn.CrossEntropyLoss()
        lr = 0.005
        fit(self.model,train_gen, verbose=1, steps_per_epoch=int(train_size // BATCH_SIZE), learning_rate=lr,loss=loss, device=self.device,change_lr_every=10)
        print("Evaluating model")
        number_images = len(test_Y)
        steps_per_epoch = int(number_images // BATCH_SIZE)
        acc, _ =evaluate(self.model,test_clean_gen, steps_per_epoch, loss,1, device=self.device)
        backdoor_acc, _ =evaluate(self.model,test_adv_gen, steps_per_epoch, loss, 1,device=self.device)
        print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


