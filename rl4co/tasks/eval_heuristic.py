import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl4co.heuristic import TabuSearch_acvrp, Random_acvrp, \
                LocalSearch2_acsp, Greedy_pg

def check_unused_kwargs(class_, kwargs):
    if len(kwargs) > 0 and not (len(kwargs) == 1 and "progress" in kwargs):
        print(f"Warning: {class_.__class__.__name__} does not use kwargs {kwargs}")



def evaluate_baseline(
    env,
    dataset_filename,
    baseline="tabu",
    save_results=True,
    save_fname="results.npz",
    **kwargs,
):
    num_loc = getattr(env, "num_loc", None)

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

    assert baseline in baselines_mapping[env.name], "baseline {} not found".format(baseline)


    # env td  data
    f = getattr(env, f"test_file") if dataset_filename is None else dataset_filename
    td_load = env.load_data(f)       # this func normalize to [0-1]

    if "sample_lst" in kwargs.keys():

        if isinstance(kwargs["sample_lst"], list):
            td_load = td_load[kwargs["sample_lst"], ...]
            print("size after sample: ", td_load["locs"].shape)

    # reset td, use this data-key-value to process in heuristic func
    td_load = env._reset(td_load) 
       
    # Set up the evaluation function
    eval_settings = baselines_mapping[env.name][baseline]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    # subsitute kwargs with the ones passed in
    kwargs_.update(kwargs)
    kwargs = kwargs_
    eval_fn = func(td_load)

    # Run evaluation
    retvals = eval_fn.forward()

    # Save results
    if save_results:
        print("Saving results to {}".format(save_fname))
        np.savez(save_fname, **retvals)
    print(f"mean reward is {retvals['mean reward']}, var is {retvals['var reward']}, time is {retvals['time']}")
    return retvals



def evaluate_baseline_withccdoadv(
    env,
    dataset_filename,
    baseline="tabu",
    adver=None,
    save_results=True,
    save_fname="results.npz",
    **kwargs,
):
    num_loc = getattr(env, "num_loc", None)

    baselines_mapping = {
        "svrp": {
            "tabu": {
                "func": TabuSearch_acvrp, "kwargs": {}},
            "random": {
                "func": Random_acvrp, "kwargs": {}
            },},
        "csp": {
            "LS2": {"func": LocalSearch2_acsp, "kwargs": {}},
        },
        "pg": {
            "LS":  {"func": Greedy_pg, "kwargs": {}},
        }
        
    }

    assert baseline in baselines_mapping[env.name], "baseline {} not found".format(baseline)


    # env td  data
    f = getattr(env, f"test_file") if dataset_filename is None else dataset_filename
    td_load = env.load_data(f)       # this func normalize to [0-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # reset td, use this data-key-value to process in heuristic func
    td_load = env._reset(td_load).to(device)

    # adver
    out_adv = adver(td_load.clone(), phase="test", return_actions=True)
    td = env.reset_stochastic_var(td_load, out_adv["action_adv"][..., None])
    
    # heuristic is faster in cpu than GPU
    td = td.to("cpu")
    # Set up the evaluation function
    eval_settings = baselines_mapping[env.name][baseline]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    # subsitute kwargs with the ones passed in
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


