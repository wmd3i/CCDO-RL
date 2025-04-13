from rl4co.model_ccdo.utils_ccdo import load_stoch_data, stochdata_key_mapping
from tensordict.tensordict import TensorDict
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from rl4co.data.dataset import tensordict_collate_fn
import torch
import numpy as np

def get_stoch_data_of_adv(stoch_data, ds_from, env, target_ds_dir, adv_idx, 
                          dataset_size, dataset_batch_size, test_dl):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ds_from == "load":
        stoch_dict, stoch_data = load_stoch_data(env, target_ds_dir, stoch_data, adv_idx)
        stoch_dict = TensorDict(stoch_dict, batch_size=dataset_size, device=device)
        stoch_dataset = TensorDictDataset(stoch_dict)
        stoch_dl = DataLoader(stoch_dataset, batch_size=dataset_batch_size, collate_fn=tensordict_collate_fn)
        save_stoch_data = False
    elif ds_from == "get_and_save":
        if adv_idx in stoch_data[stochdata_key_mapping[env.name][0]].keys():
            stoch_data_ = {}
            for sk in stochdata_key_mapping[env.name]:
                stoch_data_[sk]= stoch_data[sk][adv_idx]

            stoch_data_ = TensorDict(stoch_data_, batch_size=dataset_size, device=device)     # tensodridct的batch_size必须= 总size， dataloader可以不是
            stoch_dataset = TensorDictDataset(stoch_data_)
            stoch_dl = DataLoader(stoch_dataset, batch_size=dataset_batch_size, collate_fn=tensordict_collate_fn)
            save_stoch_data = False
        else:       # 在play_game中得到
            save_stoch_data = True
            stoch_dl = test_dl  
    
    return stoch_dl, save_stoch_data

def save_stoch_data_of_adv(ds_from, save_stoch_data, target_ds_dir, env,
                           stoch_data, adv_idx, ):
    if ds_from == "get_and_save":
        if not save_stoch_data:
            pass
        else:
            stoch_save_dir = target_ds_dir + "/" 
            stoch_data_save_pth = stoch_save_dir + "adv_"+str(adv_idx) + ".npz"       # to ssave new stoch data
            stochdata_key_lst = stochdata_key_mapping[env.name]
            if len(stochdata_key_lst) > 1:
            
                for sk in stochdata_key_lst:
                    s_pth = stoch_data_save_pth[:-4] + "_var_" + sk + ".npz"
                    print(f"save stoch_data to {s_pth}, {stoch_data[sk][adv_idx][0]}")

                    np.savez(s_pth, stoch_data[sk][adv_idx].cpu())       # 自动转化为numpy
            else:
                for sk in stochdata_key_lst:
                    print(f"save stoch_data to {stoch_data_save_pth}, {stoch_data[sk][adv_idx][0]}")
                    np.savez(stoch_data_save_pth, stoch_data[sk][adv_idx].cpu())
