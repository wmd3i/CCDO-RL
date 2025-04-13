import random
import numpy as np
import nashpy as nash
import torch
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
import hydra
from lightning import Callback, LightningModule
from rl4co.data.dataset import tensordict_collate_fn
from rl4co.heuristic import TabuSearch_acvrp, Random_acvrp, \
    LocalSearch2_acsp, Greedy_pg
from tensordict.tensordict import TensorDict


baselines_mapping = {
        "acvrp": {
            "tabu": {
                "func": TabuSearch_acvrp, "kwargs": {}},
            "random": {
                "func": Random_acvrp, "kwargs": {}
            },},
        "acsp": {
            "LS2": {"func": LocalSearch2_acsp, "kwargs": {}},
        },
        "pg": {
            "greedy_op": {"func": Greedy_pg, "kwargs": {}},
        }
        
    }

stochdata_key_mapping = {
    "acvrp": ["real_demand"],        # reset_stochastic_var()里改变的
    "acsp": ["stochastic_maxcover"],
    "pg": ["real_prob", "stochastic_cost"]   # attack_prob 不变； 随机变量tmp=real-prob, 决定stoch_cost (cost已知)
}

    



def sample_strategy(distrib):
    strategy_is = random.choices(list(range(len(distrib))), weights=distrib)
    # print("in func", strategy_is)
    strategy_i = strategy_is[0]
    return strategy_i

def nash_solver(payoff):
    """ given payoff matrix for a zero-sum normal-form game, numpy arrays
    return first mixed equilibrium (may be multiple)
    returns a tuple of numpy arrays """
    game = nash.Game(payoff)
    with np.errstate(invalid='raise'):
        equilibria = game.lemke_howson_enumeration()
        equilibrium = next(equilibria, None)

        # Lemke-Howson couldn't find equilibrium OR
        # Lemke-Howson return error - game may be degenerate. try other approaches
        if equilibrium is None or (equilibrium[0].shape != (payoff.shape[0],) and equilibrium[1].shape != (payoff.shape[0],)):
            # try other
            print('\n\n\n\n\nuh oh! degenerate solution')
            print('payoffs are\n', payoff)
            equilibria = game.vertex_enumeration()
            equilibrium = next(equilibria)
            if equilibrium is None:
                print('\n\n\n\n\nuh oh x2! degenerate solution again!!')
                print('payoffs are\n', payoff)
                equilibria = game.support_enumeration()
                equilibrium = next(equilibria)

        assert equilibrium is not None
        return equilibrium

def load_stoch_data(env, stoch_dir, stoch_data, adv_idx, sample_lst=None):
    '''
    Load random variable data from the specified adv from the specified path, store it in stoch_data, and update the data(td)
    stoch_dict: All random data from adv idx at the moment
    stoch_data: All random data from all advs
    '''
    stochdata_key_lst = stochdata_key_mapping[env.name]
    stoch_dict = {}
    stoch_pth = stoch_dir +"/"+ "adv_"+str(adv_idx) + ".npz"
    if len(stochdata_key_lst) > 1:
        for sk in stochdata_key_lst:
            s_pth = stoch_pth[:-4] + "_var_" + sk + ".npz"
            
            stoch_data_ = torch.from_numpy(dict(np.load(s_pth))["arr_0"])
            if sample_lst:
                stoch_dict[sk] = stoch_data_[sample_lst, ...]
            else:
                stoch_dict[sk] = stoch_data_
            if stoch_data != None:
                stoch_data[sk][adv_idx] = stoch_data_[sample_lst, ...]
            print(f"load stoch_data from {s_pth}, {stoch_data[sk][adv_idx][0][:5]}")
    else:
        for sk in stochdata_key_lst:
            # stoch_pth = stoch_dir + "adv_"+str(adv_idx) + ".npz"
            stoch_data_ = torch.from_numpy(dict(np.load(stoch_pth))["arr_0"])
            if sample_lst:
                stoch_dict[sk] = stoch_data_[sample_lst, ...]
            else:
                stoch_dict[sk] = stoch_data_
            if stoch_data != None:
                stoch_data[sk][adv_idx] = stoch_data_[sample_lst, ...]
    return stoch_dict, stoch_data

def update_stoch_data(env, data, new_stoch_data, adv_idx):
    '''
        data: tensordict
        new_stoch_data: TensorDict
            size: [1, data_size, num_loc]
    '''
    stochdata_key_lst = stochdata_key_mapping[env.name]
    for sk in stochdata_key_lst:
        data.set(sk, new_stoch_data[sk][adv_idx])
    return data

def update_payoff(cfg, env, data_pth, stoch_data, save_dir,
                  protagonist, adversary, payoff_prot, row_range, col_range, rewards_rl, rewards_bl, eval_baseline=False, save_pth="./"):
    '''
        row 和col的policy 进行 play_game 填充所有pair的 payoff:
        -----------------
        |xxxx | fill fill
        |-----  fill fill
        |fill fill fill

        rewards_rl, rewards_bl:  list , only append if r==0, 
                            [[rewards with adv0], 
                             [with adv1]
                             ...]
     '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # td_init = env.reset(val_data.clone()).to(device)        # 同样数据会进行多次play game，所以val_data需要保持原样，每次game：td_init重新加载

    val_data = env.load_data(data_pth)
    val_dataset = TensorDictDataset(val_data)
    val_dl = DataLoader(val_dataset, batch_size=cfg.model_ccdo.test_batch_size, collate_fn=tensordict_collate_fn)

    
    # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
    protagonist_model: LightningModule = hydra.utils.instantiate(cfg.model, env)
    # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    
    orig_r = len(payoff_prot)
    if orig_r > 0:
        orig_c = len(payoff_prot[0])
    else:
        orig_c = 0

    
    for r in row_range:
        if r > orig_r - 1:
            new_row_payoff = []
        protagonist_model.policy = protagonist.get_policy_i(r)

        

        for c in col_range:
            adversary_model.policy, adversary_model.critic = adversary.get_policy_i(c)

            eval_bl = True if r == 0 and eval_baseline else False
            baseline_fn = save_pth+"baseline_"+cfg.baseline_method+"_"+str(r)+" row_"+str(c)+" col.npz"
            rl_rewards = [] # batch iter,
            rl_rewards_all = None   # batch's size

            bl_rewards = [] # batch iter,
            bl_rewards_var = [] # batch iter,
            bl_rewards_all = None       # batch's size

            

            if c in stoch_data[stochdata_key_mapping[env.name][0]].keys():     # load saved data
                # read or load stoch data
                stoch_data_ = {}
                for sk in stochdata_key_mapping[env.name]:
                    stoch_data_[sk]= stoch_data[sk][c]
                stoch_data_ = TensorDict(stoch_data_, batch_size=cfg.model_ccdo.test_data_size, device=device)     # tensodridct的batch_size必须= 总size， dataloader可以不是
                stoch_dataset = TensorDictDataset(stoch_data_)
                stoch_dl = DataLoader(stoch_dataset, batch_size=cfg.model_ccdo.test_batch_size, collate_fn=tensordict_collate_fn)

                save_stoch_data = False
                save_stoch_pth = None
            else:
                stoch_dl = val_dl   # just for not raise error
                save_stoch_data = True
                save_stoch_pth = save_dir + "adv_"+str(c) + ".npz"
            print(f"in r {r} - c {c}")

            
            for batch, stoch_batch in zip(val_dl, stoch_dl):
                
                rl_res, bl_res, stoch_data = play_game(env, batch.clone(), stoch_batch, stoch_data, c, 
                                           protagonist_model, adversary_model, 
                                           save_stoch_data,
                                                 eval_bl, cfg.baseline_method, baseline_fn,)
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
            
            stochdata_key_lst = stochdata_key_mapping[env.name]
            if save_stoch_data:
                if len(stochdata_key_lst) > 1:
                    for sk in stochdata_key_lst:
                        s_pth = save_stoch_pth[:-4] + "_var_" + sk + ".npz"
                        print(f"save stoch_data to {s_pth}, {stoch_data[sk][c][0]}")

                        np.savez(s_pth, stoch_data[sk][c].cpu())

                else:
                    for sk in stochdata_key_lst:
                        np.savez(save_stoch_pth, stoch_data[sk][c].cpu())
                    print(f"save stoch_data to {save_stoch_pth}, {stoch_data[sk][c][0]}")
            # record rewards of rl and baseline ,only under prog 0 
            if eval_bl and r == 0:
                rewards_bl.append(bl_rewards_all.cpu().tolist())  
            if r == 0:  # 
                rewards_rl.append(rl_rewards_all.cpu().tolist())

            payoff = torch.tensor(rl_rewards).mean().item()

            if r > orig_r - 1:
                new_row_payoff.append(payoff)
            if c > orig_c -1 and r < orig_r -1:
                payoff_prot[r].append(payoff)

        if r > orig_r - 1:
            payoff_prot.append(new_row_payoff)
    return payoff_prot, rewards_rl, rewards_bl, stoch_data

def get_bs_utility(payoff, prog_strategy, adv_strategy, prog_bs=True):
    '''
    payoff: nested list, updated one
    prog_strategy: if prog bs, 

    '''
    if prog_bs:
        A = np.array(payoff)
        A = A[:, :-1]
        rps = nash.Game(A)
        result = rps[prog_strategy, adv_strategy]
        return result[0]
    else:
        A = np.array(payoff)
        A = A[:-1, :]
        rps = nash.Game(A)
        result = rps[prog_strategy, adv_strategy]
        return result[1]


def eval(payoff, prog_strategy, adver_strategy):
    '''
    According to the strategy and payoff table, get the nash result
    '''
    
    A = np.array(payoff)
    rps = nash.Game(A)
    if len(prog_strategy) != len(adver_strategy):
        print("strategy dims not equal, just eval different ccdo is ok!")
    result = rps[prog_strategy, adver_strategy]
    return result[0]

def eval_oneprog_adv(payoff, adver_strategy):

    A = np.array(payoff)
    rps = nash.Game(A)
    result = rps[1, adver_strategy]
    return result[0]

def eval_oneprog_adv_allgraph(rewards_graph, adv_strategy):
    '''
    rewards_graph: numpy array, must shape:(graph_nums, adv_policy_Num), if not transpose
    adv_strategy: numpy array, shape:(adv_policy_Num), must be 1 dim
    '''
    if not isinstance(rewards_graph, np.ndarray):
        rewards_graph = np.array(rewards_graph)
    if rewards_graph.shape[1] != len(adv_strategy):    
        rewards_graph = rewards_graph.T
    reward_adv_graphs = np.matmul(rewards_graph, np.array(adv_strategy))
    return reward_adv_graphs

def play_game(env, td_init, stoch_td, stoch_data,  adv_idx, prog, adver=None, 
              new_stoch_data=False,
              eval_baseline=False, baseline="cw", baseline_result_fn="", **kw):
    '''
        return the reward in one evaluation: prog-adver
        prog: AM model
        adver: PPOContin model

        return:
            mean_reward: scalar, float
            ret["reward"]: tensor, 1dim , batch's size

            baseline_mean: scalar, float
            baseline_var: scalar, float
            baseline_rewards: tensor, 1dim , batch's size

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(td_init).to(device)
    prog_model = prog.to(device)
    
    stochdata_key_lst = stochdata_key_mapping[env.name]
    if adver:
        if new_stoch_data:  # get stoch data form adver, save 
            adver_model = adver.to(device)
            adver_model.eval()
            out_adv = adver_model(td_init.clone(), phase="test", return_actions=True)
            td = env.reset_stochastic_var(td_init, out_adv["action_adv"][..., None])    # env transition: get new real demand
            
            for sk in stochdata_key_lst:
                if adv_idx not in stoch_data[sk].keys():
                    stoch_data[sk][adv_idx] = td[sk].clone()        #  save stochastic data, [minibatch, size 
                else:
                    stoch_data[sk][adv_idx] = torch.cat((stoch_data[sk][adv_idx], td[sk].clone()), dim=0)   # [bigbatch, size]
        else:       # load saved stoch data and reset td
            for sk in stochdata_key_lst:
                td_init.set(sk, stoch_td[sk])
            td = td_init
            print(f"load {adv_idx} stoch data: {td[stochdata_key_lst[0]][0]} from this batch")
    else:
        td = td_init.clone()
    
    prog_model.eval()
    ret = prog_model(td, phase="test")      # must test to get reproduced result in evaluation: decode_type=greedy
    mean_reward = ret["reward"].mean().item()
    # print(mean_reward)
    
    # eval baseline under same adver data
    baseline_mean, baseline_var, baseline_rewards = None, None, None
    if adver and eval_baseline:     # eval only if adver
        baseline_results = eval_baseline_by_adv_data(env, td, baseline, True, baseline_result_fn)
        baseline_mean = baseline_results["mean reward"]
        baseline_var = baseline_results["var reward"]
        baseline_rewards = baseline_results["rewards"]
        if not isinstance(baseline_rewards, torch.Tensor):      # may list or tensor in heuristic algor
            baseline_rewards = torch.tensor(baseline_rewards)

    return [mean_reward, ret["reward"]], [baseline_mean, baseline_var, baseline_rewards], stoch_data


def play_game_heuristic(env, td_init, baseline, adv_idx, stoch_td, save_result_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(td_init).to(device)

    stochdata_key_lst = stochdata_key_mapping[env.name]
    for sk in stochdata_key_lst:
        td_init.set(sk, stoch_td[sk])
        td = td_init.clone()
    print(f"load {adv_idx} stoch data: {td[stochdata_key_lst[0]][0]}")

    # eval baseline under same adver data
    baseline_mean, baseline_var, baseline_rewards = None, None, None
    baseline_results = eval_baseline_by_adv_data(env, td, baseline, True, save_result_fn)
    baseline_mean = baseline_results["mean reward"]
    baseline_var = baseline_results["var reward"]
    baseline_rewards = baseline_results["rewards"]
    if not isinstance(baseline_rewards, torch.Tensor):      # may list or tensor in heuristic algor
        baseline_rewards = torch.tensor(baseline_rewards)
    return baseline_mean, baseline_var, baseline_rewards

def eval_baseline_by_adv_data(env, td, baseline, save_results=True, save_fname="baseline_result.npz", **kwargs):
    # heuristic is faster in cpu than GPU
    td = td.to("cpu")
    # Set up the evaluation function
    eval_settings = baselines_mapping[env.name][baseline]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    kwargs_.update(kwargs)
    kwargs = kwargs_
    eval_fn = func(td)
    # Run evaluation
    retvals = eval_fn.forward()
    # Save results
    if save_results:
        print("Saving results to {}".format(save_fname))
        np.savez(save_fname, **retvals)
    print(f"mean reward is {retvals['mean reward']}, var is {retvals['var reward']} time is {retvals['time']}")

    return retvals