from Defense.Friendly_noise import *




def run_datapoison_reinforce(model=None,method='FriendlyNoise',train_dataloader=None, params=None):
    if method=='FriendlyNoise':
        reinforced_dataset = reinforce_dataset(path="./Friendly_noise_data/", tag="demoCIFAR10", load=False, **friendly_noise_params)
