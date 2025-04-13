from typing import List, Optional, Tuple
from rl4co.tasks.eval_ccdo import evaluate_psro_policy

import hydra
import lightning as L
import pyrootutils
import torch
import collections
from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy
from rl4co.data.generate_data import generate_default_datasets, generate_dataset
from rl4co.model_adversary import PPOContiAdvModel
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from rl4co.data.dataset import tensordict_collate_fn
from rl4co.tasks.train_ccdo import update_payoff, nash_solver, play_game, eval, sample_strategy, eval_noadver, play_game_heuristic
from rl4co.tasks.train_ccdo import Adversary, Protagonist
from rl4co import utils
from rl4co.utils import RL4COTrainer
import numpy as np
import random
import nashpy as nash
import os
import time
from rl4co.model_ccdo.utils_ccdo import eval_oneprog_adv, eval_oneprog_adv_allgraph, stochdata_key_mapping, load_stoch_data
from tensordict.tensordict import TensorDict
from rl4co.utils.dataset_utils import get_stoch_data_of_adv, save_stoch_data_of_adv
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)




@utils.task_wrapper
def eval_withccdoadv(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    # trainer.logger = logger
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    data_cfg = {
            "val_data_size": cfg.model_ccdo.val_data_size,
            "test_data_size": cfg.model_ccdo.test_data_size,
        }
    generate_default_datasets(data_dir=cfg.paths.data_dir, data_cfg=data_cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 改：分别构建 prog, adv的算法model，for r c 遍历只需要重新load网络model
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    protagonist_model = protagonist_model.to(device)
    # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    adversary_model = adversary_model.to(device)
    

    if cfg.get("eval_otherprog_with_psroadv"):
        # load psro prog and adver
        
        protagonist_tmp = Protagonist(AttentionModel, AttentionModelPolicy, env)
        protagonist_tmp.load_model_weights(cfg.evaluate_adv_dir+"/models_weights/")
        adversary_tmp = Adversary(PPOContiAdvModel, PPOContiAdvPolicy, CriticNetwork, env)
        adversary_tmp.load_model_weights(cfg.evaluate_adv_dir+"/models_weights")

        data = np.load(cfg.adv_npz_pth)  # 加载
        adver_strategy = data['adver_strategy']

        prog_strategy = data['prog_strategy']

        # load test/val data
        if cfg.env.eval_dataset == "test":
            test_data_pth = cfg.env.data_dir+"/"+cfg.env.test_file
            dataset_size = cfg.model_ccdo.test_data_size
            dataset_batch_size = cfg.model_ccdo.test_batch_size
        elif cfg.env.eval_dataset == "val":
            test_data_pth = cfg.env.data_dir+"/"+cfg.env.val_file
            dataset_size = cfg.model_ccdo.val_data_size
            dataset_batch_size = cfg.model_ccdo.val_batch_size

        print(f"get eval data from {test_data_pth}")
        test_data = env.load_data(test_data_pth)

        ds_dirs = os.listdir(cfg.evaluate_adv_dir)
        target_d = "adv_stoch_data_" + cfg.env.dataset_flag 
        target_ds_dir = cfg.evaluate_adv_dir + "/"+ target_d

        if target_d in ds_dirs:
            ds_from = "load"
            if cfg.env.dataset_state == "sample":
                sample_lst  = dict(np.load(target_ds_dir+".npz"))["sample_lst"]
                print(f"sample from {sample_lst}")
                test_data = test_data[sample_lst, ...]
                dataset_size = 100
        else:
            ds_from = "get_and_save"
            if cfg.env.dataset_state == "sample":
                sample_lst = random.choices(range(test_data["locs"].shape[0]), k=100)
                np.savez(target_ds_dir, sample_lst=sample_lst)
                print("sample : ", sample_lst)
                test_data = test_data[sample_lst, ...]
                print("size after sample: ", test_data["locs"].shape)
                dataset_size = 100
            else:
                sample_lst = None

        if ds_from == "load":
            pass
        elif ds_from == "get_and_save":
            assert not os.path.exists(target_ds_dir), "dataset dir has exists!"
            os.makedirs(target_ds_dir)

        stoch_data = {}
        for sk in stochdata_key_mapping[env.name]:
            stoch_data[sk] = {}

        if cfg.get("eval_rl_prog"):
            # 
            print("eval rl agent with ccdo-adversary on ", test_data_pth)
            
            rl_prog_pth = cfg.rl_prog_pth
            protagonist_model = protagonist_model.load_from_checkpoint(rl_prog_pth)
            st = time.time()
            length = min(adversary_tmp.policy_number, len(adver_strategy))
            rewards_rl = []

            for c in range(length):

                adversary_model.policy, adversary_model.critic = adversary_tmp.get_policy_i(c)
                # The same data will be played multiple times, so val_data will need to be left intact and reloaded each time game
                rl_rewards = [] # [batch iter,]
                rl_rewards_all = None   # 1dim, [data_size's size

                bl_rewards = [] # batch iter,
                bl_rewards_var = [] # batch iter,
                bl_rewards_all = None       # batch's size

               

                test_data_ = test_data.to(device)
                test_dataset = TensorDictDataset(test_data_)
                test_dl = DataLoader(test_dataset, batch_size=dataset_batch_size, collate_fn=tensordict_collate_fn)


                
                stoch_dl, save_stoch_data = get_stoch_data_of_adv(stoch_data, ds_from, env, target_ds_dir, c, dataset_size, dataset_batch_size, test_dl)

                for batch, stoch_batch in zip(test_dl, stoch_dl):
                    if ds_from == "load":
                        rl_res, bl_res, stoch_data = play_game(env, batch.clone(), stoch_batch, stoch_data, c, 
                                                protagonist_model, adversary_model, False, "", False,)
                    elif ds_from == "get_and_save":
                        if c in stoch_data[stochdata_key_mapping[env.name][0]].keys():
                            rl_res, bl_res, stoch_data = play_game(env, batch.clone(), stoch_batch, stoch_data, c, 
                                                protagonist_model, adversary_model, False, "", False,)
                        else:
                            rl_res, bl_res, stoch_data = play_game(env, batch.clone(), None, stoch_data, c, 
                                                    protagonist_model, adversary_model, True, False)
                    
                    batch_rl_mean, batch_rl_allg = rl_res
                    batch_l_mean, batch_bl_var, batch_bl_allg = bl_res

                    rl_rewards.append(batch_rl_mean)     
                    if rl_rewards_all == None:
                        rl_rewards_all = batch_rl_allg
                    else:
                        rl_rewards_all = torch.cat((rl_rewards_all, batch_rl_allg), dim=0)
                    
                    bl_rewards.append(batch_l_mean)
                    bl_rewards_var.append(batch_bl_var)
                    if bl_rewards_all == None:
                        bl_rewards_all = batch_bl_allg
                    else:
                        bl_rewards_all = torch.cat((bl_rewards_all, batch_bl_allg), dim=0)
                
                save_stoch_data_of_adv(ds_from, save_stoch_data, target_ds_dir, env,
                                       stoch_data, c)

                rewards_rl.append(rl_rewards_all.cpu().tolist())
            rl_rewards_psro = eval_oneprog_adv_allgraph(rewards_rl, adver_strategy)
            rl_mean, rl_var = rl_rewards_psro.mean(), rl_rewards_psro.var()

            eval_time = time.time() - st
            print(f"reward mean is {rl_mean}, var is {rl_var}, eval time of rl is {eval_time} s")

            save_eval_pth = "eval_rl_"+cfg.env.dataset_flag+"_withadv.npz"

            np.savez(cfg.evaluate_adv_dir+ '/'+save_eval_pth, 
                        rl_pth=rl_prog_pth,
                        eval_reward=rl_mean,
                        eval_var=rl_var,
                        eval_time=eval_time,
                        eval_all_r=rewards_rl,
                        eval_payoffs=rl_rewards_psro,       # key=value
                        eval_data=test_data_pth,
                        eval_adver_strategy=adver_strategy)  # 保
        
        payoff_underadv_baseline = []
        if cfg.get("eval_baseline_prog"):
            
            prog_baseline = cfg.baseline_heur
            print(f"eval {prog_baseline} baseline with ccdo-adversary")

            save_bl_res_pth = cfg.evaluate_adv_dir+ '/baseline_'+prog_baseline
            if not os.path.exists(save_bl_res_pth):
                os.mkdir(save_bl_res_pth)

            support_adv_stra_idx = [i for i, x in enumerate(adver_strategy) if x>0]
            support_adv_stra = [x for i, x in enumerate(adver_strategy) if x>0]
            print(support_adv_stra)

            stochdata_key_lst = stochdata_key_mapping[env.name]

            st = time.time()
            rewards_bl = []
            for adv_idx in support_adv_stra_idx:
                
                test_data_ = test_data.to(device)
                test_dataset = TensorDictDataset(test_data_)
                test_dl = DataLoader(test_dataset, batch_size=cfg.model_ccdo.test_batch_size, collate_fn=tensordict_collate_fn)

                if adv_idx in stoch_data[stochdata_key_lst[0]]:     # no load again
                    stoch_dict = {}
                    for sk in stochdata_key_lst:
                        stoch_dict[sk] = stoch_data[sk][adv_idx]
                    save_stoch_data = False
                else:
                        
                    stoch_dl, save_stoch_data = get_stoch_data_of_adv(stoch_data, ds_from, env, target_ds_dir, adv_idx, dataset_size, dataset_batch_size, test_dl)

                bl_rewards = [] # batch iter,
                bl_rewards_var = [] # batch iter,
                bl_rewards_all = None       # batch's size
                for batch, stoch_batch in zip(test_dl, stoch_dl):

                    batch_l_mean, batch_bl_var, batch_bl_allg = play_game_heuristic(env, batch.clone(), 
                                                                                    prog_baseline, adv_idx, 
                                                                                    stoch_batch, save_bl_res_pth)
                    bl_rewards.append(batch_l_mean)
                    bl_rewards_var.append(batch_bl_var)
                    if bl_rewards_all == None:
                        bl_rewards_all = batch_bl_allg
                    else:
                        bl_rewards_all = torch.cat((bl_rewards_all, batch_bl_allg), dim=0)
                rewards_bl.append(bl_rewards_all.cpu().tolist())

                save_stoch_data_of_adv(ds_from, save_stoch_data, target_ds_dir, env, stoch_data,
                                       adv_idx,)
                
            eval_time = time.time()-st 
            bl_rewards_psro = eval_oneprog_adv_allgraph(rewards_bl, support_adv_stra)
            bl_mean, bl_var = bl_rewards_psro.mean(), bl_rewards_psro.var()
            print(f"baseline: reward mean is {bl_mean}, var is {bl_var}, eval time of baseline:{prog_baseline} is {eval_time} s")
            save_eval_pth = "eval_baseline_"+prog_baseline+"_"+cfg.env.dataset_flag+"_withadv.npz"

            np.savez(cfg.evaluate_adv_dir+ '/'+save_eval_pth, 
                    baseline=prog_baseline,
                    eval_reward=bl_mean,
                    var_eval=bl_var,
                    eval_time=eval_time,
                    eval_payoffs=bl_rewards_psro,       # key=value
                    eval_data=test_data_pth,
                    eval_adver_strategy=adver_strategy,
                    support_adv=support_adv_stra,
                    support_adv_idx=support_adv_stra_idx)  # 保
    return None, None
    






@hydra.main(version_base="1.3", config_path="../../configs", config_name="main_ccdo_frame.yaml")
def eval_other_withccdo_adv(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    eval_withccdoadv(cfg)


if __name__ == "__main__":
    eval_other_withccdo_adv()
