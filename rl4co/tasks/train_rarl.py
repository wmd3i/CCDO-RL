from typing import List, Optional, Tuple
from rl4co.tasks.eval_rarl import evaluate_rarl_policy

import hydra
import lightning as L
import pyrootutils
import torch

from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rl4co import utils
from rl4co.utils import RL4COTrainer
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)


@utils.task_wrapper
# @profile(stream=open('log_mem_cvrp50.log', 'w+'))
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # h = hpy().heap()
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    
        
    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    
    if cfg.fix_graph:
        graph_pool = hydra.utils.instantiate(cfg.graph_pool)
        graph_pool.generate_datas()
        env.get_fix_data(graph_pool)
    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
    protagonist: LightningModule = hydra.utils.instantiate(cfg.model, env)
    
    if cfg.load_prog_from_path:
        protagonist = protagonist.load_from_checkpoint(cfg.load_prog_from_path)
        print("protagonist is loaded!")
        
    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
    adversary: LightningModule = hydra.utils.instantiate(cfg.model_adversary, env)
    
    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating multi-agent model <{cfg.model_rarl._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model_rarl, env, protagonist, adversary)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "protagonist": protagonist,
        "adversary": adversary,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        if cfg.train_with_pretrain:
            model = model.load_from_checkpoint(cfg.train_with_pretrain)
            print("load rarl pretrained")
        else:
            print("no load rarl pretrained")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    
    # 额外增加evaluate 过程
    if cfg.get("evaluate"):
        method = cfg.evaluate_method
        log.info(f"Start evaluation by {method}!")
        
        if cfg.get("mode") == "evaluate":
            ckpt_rarl_path = cfg.get("ckpt_rarl_path")      # for adv
            rarl_model = model.load_from_checkpoint(ckpt_rarl_path)
            
            dataset = env.dataset(phase="test", batch_size=cfg.model_rarl.test_data_size)     # 使用test的数据集做evaluation
        
            
            change_data_model = "_nochange_" # if change env model
            withadv = "adv"
            if cfg.get("eval_withadv"):
                adv = rarl_model.adversary
                
            else:
                adv = None
                withadv = "noadv"
                
            
            save_fname_lst = []
            save_pt_lst = []
            policy_lst = []
            if cfg.get("eval_rarl"):
                # prog_ = "rarlprog"
                
                if cfg.get("another"):
                    from ..model_ccdo.am_amppo import AM_PPO
                    model_tmp = AM_PPO(env, protagonist, adversary)
                    ckpt_rarl_prog_path = cfg.get("another_rarl_prog_path")
                    model_rarl_tmp = model_tmp.load_from_checkpoint(ckpt_rarl_prog_path)
                    evaluate_model = model_rarl_tmp.protagonist
                else:
                    evaluate_model = rarl_model.protagonist
                
                policy_lst.append(evaluate_model.policy)
                save_fname = cfg.get("evaluate_loc") + "/evalu_"+withadv+"_rarlprog_"+method+change_data_model+".npz"
                save_fname_lst.append(save_fname)
                save_pt = './graphs/graph_of_'+str(env.name)+withadv+"_rarlprog_"+change_data_model+".png"
                save_pt_lst.append(save_pt)
                # evaluate_rarl_policy(env, evaluate_model.policy, adv, dataset, method, save_results=True, save_fname=save_fname, 
                #              save_pt = save_pt)

            if cfg.get("eval_rl"):
                ckpt_prog_path = cfg.get("ckpt_prog_path")
                evaluate_model_tmp = model.protagonist.load_from_checkpoint(ckpt_prog_path)
                
                policy_lst.append(evaluate_model_tmp.policy)
                save_fname_tmp = cfg.get("evaluate_loc") + "/evalu_"+withadv+"_rlprog_"+method+change_data_model+".npz"
                save_fname_lst.append(save_fname_tmp)
                save_pt_tmp = './graphs/graph_of_'+str(env.name)+withadv+"_rlprog_"+change_data_model+".png"
                save_pt_lst.append(save_pt_tmp)
                # evaluate_rarl_policy(env, evaluate_model_tmp.policy, adv, dataset, method, save_results=True, save_fname=save_fname, 
                #              save_pt = './graphs/graph_'+withadv+"_rlprog_"+change_data_model+".png")

            evaluate_rarl_policy(env, policy_lst, adv, dataset, method, save_results=True, save_fname=save_fname_lst, 
                         save_pt = save_pt_lst)
        elif cfg.get("mode") == "train":
            ckpt_path = trainer.checkpoint_callback.best_model_path
            evaluate_model = trainer.model.load_from_checkpoint(ckpt_path)
           
            save_fname = logger[0].save_dir + "/evalu_"+method+".npz"
        # log.info(f"Best ckpt path: {ckpt_rarl_path}")
        
        
            evaluate_rarl_policy(env, evaluate_model.policy, adv, dataset, method, save_results=True, save_fname=[save_fname], 
                                save_pt = ['./graphs/graph_of_'+str(env.name)+withadv+"_rarlprog_"+change_data_model+".png"])

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main_rarl.yaml")
# @hydra.main(version_base="1.3", config_path="../../configs", config_name="experiment/routing/am-ppo.yaml")
# @hydra.main(version_base="1.3", config_path="configs", config_name="experiment/routing/am-ppo.yaml")
def train_rarl(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    print("this is in rarl train")
    utils.extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    train_rarl()
