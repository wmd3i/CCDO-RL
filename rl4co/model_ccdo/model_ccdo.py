import torch
from rl4co.model_ccdo.utils_ccdo import *
import collections
import os
from rl4co import utils
from lightning import Callback, LightningModule
import hydra
from rl4co.utils import RL4COTrainer

log = utils.get_pylogger(__name__)
class Protagonist:
    def __init__(self, model, policy, env) -> None:
        self.model = model      # AttentionModel class
        self.policy = policy
        self.env = env
        self.policies = []
        self.correspond_baseline = []
        self.strategy = []
        self.no_zeroth = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_a_policy(self):
        return self.policy(self.env.name)
    
    def get_a_model(self):
        return self.model(self.env)
        
    def add_policy(self, policy):
        self.policies.append(policy)
    
    def get_policy_i(self, idx):
        assert idx > -1 and idx < self.policy_number, "idx exceeds range"
        assert type(idx) == int, "idx wrong type!"
        return self.policies[idx]
    
    def get_curr_policy(self):
        '''
        sample a policy from strategy
        '''
        sample_i = sample_strategy(self.strategy)
        sampled_policy = self.get_policy_i(sample_i)
        print(f"--- sample prog policy: {sample_i}")

        # copy the parameter and create another instance
        curr_policy = self.get_a_policy()
        state_dict = sampled_policy.state_dict()
        weight_keys = list(state_dict.keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            fed_state_dict[key] = state_dict[key].clone().to(self.device)
        curr_policy.load_state_dict(fed_state_dict)

        return curr_policy.to(self.device)
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    @property
    def strategy_length(self):
        return len(self.strategy)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def save_model_weights(self, pth_dir='./'):
        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        for i in range(len(self.policies)):
            torch.save(self.policies[i].state_dict(), f=pth_dir+"progPolicy_"+str(i)+".pth")
    
    def save_a_model_weights(self, pth_dir, i, policy):
        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        torch.save(policy.state_dict(), f=pth_dir+"progPolicy_"+str(i)+".pth")
    

    def load_model_weights(self, load_dir):
        
        models = os.listdir(load_dir)
        log.info(f"{len(models)} policies to be loaded now.")
        assert len(self.policies) == 0, "polices are not empty but load more polices!"
        length = range(len(models))
        pth_0 = load_dir+"progPolicy_"+str(0)+".pth"
        if not os.path.exists(pth_0):
            print(f" 0 not exists")
            self.no_zeroth = True
            length = range(1, len(models)+1)

        for i in length:
            pth = load_dir+"progPolicy_"+str(i)+".pth"
            model_w = torch.load(pth)
            tmp_policy = self.policy(self.env.name)
            tmp_policy.load_state_dict(model_w)
            
            self.policies.append(tmp_policy)
        

    def get_best_response(self, adversary, cfg, callbacks, logger, epoch, init=False):
        '''
        fix adversary and update Protagonist
        '''
        print("===== in protagonist bs ====")
        
        # max_epoch = 1
        if epoch == 0:
            max_epoch = cfg.prog_epoch1  #20 # 
        elif epoch > 0 and epoch < 3:
            max_epoch = cfg.prog_epoch2  #20  # 
        else: 
            max_epoch = cfg.prog_epoch3
        
        # get protagonist's policy from strategy: adver用策略变化，prog更新policy
        cur_policy = self.get_curr_policy()     # sample a AttentionModel's policy
        # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env, policy=cur_policy)
        cur_model.baseline.with_adv = True
        cur_model = cur_model.to(self.device)
        # get adver's policy from its' strategy: 
        adver_curr_policy, adver_curr_critic = adversary.get_curr_policy()
        # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        adver_cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                            policy=adver_curr_policy, critic=adver_curr_critic)
        adver_cur_model = adver_cur_model.to(self.device)
        # fix adver's params / not update
        # run until can't get more reward / reward converge
        ccdo_model: LightningModule = hydra.utils.instantiate(cfg.model_ccdo, self.env, cur_model, adver_cur_model, 
                                                            fix_protagonist=False,
                                                            fix_adversary=True,
                                                            prog_polices=self.policies,
                                                            adver_polices_and_critics=[adversary.policies, adversary.correspond_critic],
                                                            prog_strategy=self.strategy,
                                                            adver_strategy=adversary.strategy)
        log.info(f"Instantiating trainer in prog bs ...")
        trainer: RL4COTrainer = hydra.utils.instantiate(
            cfg.trainer,
            max_epochs=max_epoch,
            callbacks=callbacks,
            logger=logger,
        )

        if cfg.get("train"):
            # log.info("Starting training!")
            if init:
                if cfg.load_prog_from_path:
                    cur_model = cur_model.load_from_checkpoint(cfg.load_prog_from_path)
                    cur_policy = cur_model.policy
                    print("load ccdo pretrained")
                else:
                    trainer.fit(model=ccdo_model, ckpt_path=cfg.get("ckpt_path"))
            else:
                trainer.fit(model=ccdo_model, ckpt_path=cfg.get("ckpt_path"))

            curr_reward = ccdo_model.last_val_reward.to("cpu")

        return ccdo_model.protagonist.policy, curr_reward


class Adversary:
    def __init__(self, model, policy, critic, env) -> None:
        self.model = model
        self.policy = policy
        self.critic = critic
        self.env = env
        self.policies = []
        self.correspond_critic = []
        self.strategy = []
        self.no_zeroth = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_a_policy(self):
        return self.policy(self.env.name), self.critic(self.env.name)
    

    def get_a_model(self, opponent):
        return self.model(self.env, opponent=opponent)
    
    def add_policy(self, policy, critic):
        self.policies.append(policy)
        self.correspond_critic.append(critic)
    
    def get_policy_i(self, idx):
        assert idx > -1 and idx < self.policy_number, "idx exceeds range"
        assert type(idx) == int, "idx wrong type!"
        return self.policies[idx], self.correspond_critic[idx]
    
    def get_curr_policy(self):
        '''
        sample a policy from strategy
        '''
        sample_i = sample_strategy(self.strategy)
        sampled_policy, sampled_critic = self.get_policy_i(sample_i)
        print(f"--- sample afversary policy: {sample_i}")
        curr_policy, curr_critic = self.get_a_policy()
        # worker_state_dict = [x.state_dict() for x in self.policies]
        state_dict_policy = sampled_policy.state_dict()
        weight_keys_policy = list(state_dict_policy.keys())
        fed_state_dict_policy = collections.OrderedDict()

        for key in weight_keys_policy:
            fed_state_dict_policy[key] = state_dict_policy[key].clone().to(self.device)
        curr_policy.load_state_dict(fed_state_dict_policy)

        state_dict_critic = sampled_critic.state_dict()
        weight_keys_critic = list(state_dict_critic.keys())
        fed_state_dict_critic = collections.OrderedDict()
        for key in weight_keys_critic:
            fed_state_dict_critic[key] = state_dict_critic[key].clone().to(self.device)
        curr_critic.load_state_dict(fed_state_dict_critic)

        return curr_policy.to(self.device), curr_critic.to(self.device)
    
    @property
    def policy_number(self):
        return len(self.policies)
    
    @property
    def strategy_length(self):
        return len(self.strategy)
    
    def update_strategy(self, strategy):
        assert sum(strategy) < 1. + 1e-5, "strategy prob is wrong!"
        assert len(strategy) == self.policy_number, "strategy size not equal to policies"
        if not isinstance(strategy, list):
            strategy = strategy.tolist()
        self.strategy = strategy
    
    def save_model_weights(self, pth_dir='./'):

        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        pth_dir_policy = pth_dir+"_policy/"
        if not os.path.exists(pth_dir_policy):
            os.mkdir(pth_dir_policy)

        pth_dir_critic = pth_dir+"_critic/"
        if not os.path.exists(pth_dir_critic):
            os.mkdir(pth_dir_critic)

        for i in range(len(self.policies)):
            torch.save(self.policies[i].state_dict(), f=pth_dir_policy+"adverPolicy_"+str(i)+".pth")
            torch.save(self.correspond_critic[i].state_dict(), f=pth_dir_critic+"adverCritic_"+str(i)+".pth")
    
    def save_a_model_weights(self, pth_dir, i, policy, critic):

        if not os.path.exists(pth_dir):
            os.mkdir(pth_dir)

        pth_dir_policy = pth_dir+"_policy/"
        if not os.path.exists(pth_dir_policy):
            os.mkdir(pth_dir_policy)

        pth_dir_critic = pth_dir+"_critic/"
        if not os.path.exists(pth_dir_critic):
            os.mkdir(pth_dir_critic)

        torch.save(policy.state_dict(), f=pth_dir_policy+"adverPolicy_"+str(i)+".pth")
        torch.save(critic.state_dict(), f=pth_dir_critic+"adverCritic_"+str(i)+".pth")
    

    def load_model_weights(self, load_dir):
        # 加载到policies中
        load_dir_policy = load_dir + "_policy/"
        policies = os.listdir(load_dir_policy)
        log.info(f"{len(policies)} policies to be loaded now.")
        # assert len(self.policies) == 0 and len(self.correspond_critic) == 0, "adver polices are not empty but load more polices!"
        if not len(self.policies) == 0:
            print("reload adv from ", load_dir)
            self.policies = []
            self.correspond_critic = []

        self.no_zeroth = False  # 每次初始化为false
        length = range(len(policies))
        pth_0 = load_dir_policy+"adverPolicy_"+str(0)+".pth"
        if not os.path.exists(pth_0):
            print(f"no 0 to load")
            self.no_zeroth = True
            length = range(1, len(policies)+1)

        for i in length:
            pth = load_dir_policy+"adverPolicy_"+str(i)+".pth"
            policy_w = torch.load(pth)
            tmp_policy = self.policy(self.env.name)
            tmp_policy.load_state_dict(policy_w)
            self.policies.append(tmp_policy)
        print(f"adver num is {self.policy_number}")
        load_dir_critic = load_dir + "_critic/"
        for i in length:
            pth = load_dir_critic+"adverCritic_"+str(i)+".pth"
            critic_w = torch.load(pth)
            tmp_critic = self.critic(self.env.name)
            tmp_critic.load_state_dict(critic_w)
            self.correspond_critic.append(tmp_critic)

    def get_best_response(self, protagonist, cfg, callbacks, logger, epoch):
        '''
        fix Protagonist and update adversary
        '''
        print("===== in adversary bs ====")
        if epoch == 0:
            max_epoch = cfg.adver_epoch1  #20 # 
        elif epoch > 0 and epoch < 3:
            max_epoch = cfg.adver_epoch2  #20  # 
        else: 
            max_epoch = cfg.adver_epoch3
        # get protagonist's policy from strategy: params add
        prog_policy = protagonist.get_curr_policy()
        # log.info(f"Instantiating protagonist model <{cfg.model._target_}>")
        prog_model: LightningModule = hydra.utils.instantiate(cfg.model, self.env, policy=prog_policy)
        prog_model.baseline.with_adv = True
        prog_model = prog_model.to(self.device)
        # get adver's policy from its' strategy: params add
        cur_policy, cur_critic = self.get_curr_policy()     # PPOContinous's policy
        # log.info(f"Instantiating adversary model <{cfg.model_adversary._target_}>")
        cur_model: LightningModule = hydra.utils.instantiate(cfg.model_adversary, self.env, 
                                                             policy=cur_policy, critic=cur_critic)
        cur_model = cur_model.to(self.device)


        # fix prog's params / not update
        # run until can't get more reward / reward converge
        ccdo_model: LightningModule = hydra.utils.instantiate(cfg.model_ccdo, self.env, prog_model,
                                                              cur_model, 
                                                              fix_protagonist=True,
                                                              fix_adversary=False,
                                                              prog_polices=protagonist.policies,
                                                            adver_polices_and_critics=[self.policies, self.correspond_critic],
                                                            prog_strategy=protagonist.strategy,
                                                            adver_strategy=self.strategy)

        
        if cfg.get("train"):
            log.info(f"Instantiating trainer in adver bs ...")
            trainer: RL4COTrainer = hydra.utils.instantiate(
                    cfg.trainer,
                    max_epochs=max_epoch,
                    callbacks=callbacks,
                    logger=logger,
                )
            
            # log.info("Starting training!")
            
            trainer.fit(model=ccdo_model, ckpt_path=cfg.get("ckpt_path"))
            curr_reward = ccdo_model.last_val_reward.to("cpu")
        
        

        return ccdo_model.adversary.policy, ccdo_model.adversary.critic, curr_reward