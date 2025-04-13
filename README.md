# Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?
This repository is the official implementation of "[Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?](https://arxiv.org/abs/2502.01252)"
## Installation
First, install libraries via requirements.txt:
```bash
pip install -r requirements.txt
pip install -e . 
```
     
## Run the code
### Training
To train the model(s): protagonist, adversary, CCDO algor in the paper, following this order:
#### 1. Train Protagonist model:
```bash
python run.py
```
default experiment settings are in `configs/new_experiment/routing/` like am.yaml (all but except am-ppo_adv.yaml)

You can change to other envs and different numbers of nodes：
```bash
python run.py env=acvrp env.num_loc=20
python run.py env=acsp env.num_loc=20
python run.py env=pg env.num_loc=20
```
   
#### 2. Train adversarial model:

**You need to train a protagonist model first.** Then add its model checkpoint pth to `prog_pth`  in `routing/am-ppo_adv.yaml`. 
> Tips:  
>> checkpoint pth is in `{root_dir}/logs/train/runs/{env.name}{env.num}{env.num_loc}/am-{env.num}{env.num_loc}/{time}/rl4co/xxxxxxx/checkpoints/xxx.ckpt`  
>>> e.g.: `{root_dir}/acces_games/logs/train/runs/acsp50/am-acsp50/2024-05-22_19-57-54/rl4co/c1b28edi/checkpoints/epoch=0-step=20.ckpt`
Make sure its env and num_loc is same as protagonist.

run with command:
```bash
python run_adv.py
```
> Tips:  
> adversary model checkpoint pth is like:
>  `{root_dir}/acces_games/logs/train/runs/acsp50/ppo-adv-acsp50/2024-05-22_19-58-41/rl4co/v2jo88ct/checkpoints/epoch=0-step=40.ckpt`  
#### 3. Train CCDO-RL model:

**After train protagonist and adversarial, you can run CCDO-RL now.**  
Set protagonist  and adversarial checkpoint pth in `load_prog_from_path` and `load_adv_from_path` in`new_experiment/CCDO/ccdo_am-ppo.yaml`
Make sure their env and num_loc is same as protagonist & adversary.
```bash
python run_ccdo.py
```
Now you get a ccdo-protagonist and a ccdo-adversary.

### Evaluation
#### Datasets
if evaluate on sampled ones from trained dataset, set cooresponding params in `configs/env/{env_name}.yaml`
```bash
eval_dataset: "val"    
dataset_state: "sample"
```
If on test dataset, set:
```bash
eval_dataset: "test"    
dataset_state: "no_sample"
```


#### Eval trained RL agent

##### 1. Eval protagonist without adversary:

Firstly, set params `ckpt_path` to trained protagonist checkpoint path(ckpt) and its  dir to `evaluate_savedir` in `greedy_eval.yaml`. e.g.
```bash
evaluate_savedir: {root_dir}/acces_games/logs/train/runs/acsp50/am-acsp50/2024-05-22_19-57-54  
ckpt_path: {root_dir}/acces_games/logs/train/runs/acsp50/am-acsp50/2024-05-22_19-57-54/rl4co/c1b28edi/checkpoints/epoch=0-step=20.ckpt

```
Then run the commend:
```bash
python run.py evaluate=greedy_eval
```
You can find detailed results in the protagonist logdir.
> Tips:  
> If you eval on sampled train dataset,  you must run an CCDO algorithm and eval any protagonist(is or not ccdo) on it firstly. Except for `evaluate_savedir` and `ckpt_path`, also set `evaluate_psro_dir` to ccdo alogr directory.
>

##### 2. Eval ccdo-protagonist with ccdo-adversary:

Set `evaluate_prog_dir` to ccdo logdir.
Set `eval_withadv` to true.
Run:
```bash
python run_ccdo.py evaluate=greedy_eval_ccdo
```
> Tips:  
> Must do this firstly if evaluate any with ccdo-adversary.

##### 3. Eval ccdo-protagonsit without adversary:
Following last one, set  'eval_withadv` to false.
Still run:
```bash
python run_ccdo.py evaluate=greedy_eval_ccdo
```
##### 4. Eval protagonist with ccdo-adversary:

Modify `evaluate` in main_ccdo_frame.yaml as `eval_other_with_ccdo_adver.yaml` firstly.
In `eval_other_with_ccdo_adver.yaml`, set `evaluate_adv_dir` to the ccdo logdir path, and `eval_rl_prog` to `true`.
Set 'rl_prog_dir' and 'rl_prog_pth' to trained protagonist dir and ckpt

Then run the commend:
```bash
python run_ccdoadv_eval.py
```







#### Eval heuristic algorithm:
##### 1. Eval heuristic-alg without adversary:


Firstly, set params `ckpt_path` to trained protagonist checkpoint path(ckpt) and its  dir to `evaluate_savedir` in `baseline_eval.yaml`. e.g.
Select the baseline method in `baseline`.
Then run the commend:
```bash
python run.py evaluate=baseline_eval
```
##### 2. Eval heuristic-alg with ccdo-adversary:

Modify `evaluate` in main_ccdo_frame.yaml as `eval_other_with_ccdo_adver.yaml` firstly.
In `eval_other_with_ccdo_adver.yaml`, set `evaluate_adv_dir` to the ccdo logdir path, and `eval_baseline_prog` to `true`.
Set 'rl_prog_dir' and 'rl_prog_pth' to trained protagonist dir and ckpt
Select the baseline method in `baseline_heur`.
Then run the commend:
```bash
python run_ccdoadv_eval.py
```
