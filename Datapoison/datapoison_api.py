from Datapoison.Defense.Friendly_noise import *
from Datapoison.Attack.base import DatapoisonAttack


def run_datapoison_reinforce(model=None, method='FriendlyNoise', train_dataloader=None, params=None):
    if method == 'FriendlyNoise':
        reinforced_model_path,Reinforced_ACC = datapoison_model_reinforce(model=model, dataloader=train_dataloader, params=params)
        return reinforced_model_path,{"After_Datapoison_Defense_ACC":Reinforced_ACC}


def run_datapoison_test(model=None, method='forest', train_dataloader=None, params=None):
    if method == 'forest':
        target = DatapoisonAttack(local_model=model, scenario="from_scratch", epochs=10, attack_iter=250, restarts=1)
        datapoisonrst = target.test()

    return datapoisonrst
