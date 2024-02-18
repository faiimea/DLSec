from Datapoison.Defense.Friendly_noise import *
from Datapoison.Attack.base import DatapoisonAttack


def run_datapoison_reinforce(model=None, method='FriendlyNoise', train_dataloader=None, params=None):
    if method == 'FriendlyNoise':
        reinforced_model_path,Reinforced_ACC = datapoison_model_reinforce(model=model, dataloader=train_dataloader, params=params)
        return reinforced_model_path,{"After_Datapoison_Defense_ACC":Reinforced_ACC}


def run_datapoison_test(model=None, method='poison-frogs', train_dataloader=None, params=None):
    if method == 'poison-frogs':
        target = DatapoisonAttack(params)
        datapoisonrst = target.test()

    return datapoisonrst
