# Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?
This repository is the official implementation of "[Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?](https://arxiv.org/abs/2502.01252)", including the **CCDO-RL** algorithm and heuristic baselines.

**Authors: Yuheng Li, Panpan Wang, Haipeng Chen**

**Accepted by ICLR 2025**

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
>> checkpoint pth is in `{root_dir}/logs/train/runs/{env.name}{env.num_loc}/am-{env.num}{env.num_loc}/{time}/rl4co/xxxxxxx/checkpoints/xxx.ckpt`  
Make sure its env and num_loc are the same as those of the protagonist.

run with command:
```bash
python run_adv.py
```

#### 3. Train CCDO-RL model:

**After train protagonist and adversarial, you can run CCDO-RL now.**  
Set protagonist  and adversarial checkpoint pth in `load_prog_from_path` and `load_adv_from_path` in`new_experiment/CCDO/ccdo_am-ppo.yaml`
Make sure their env and num_loc are the same as those of protagonist & adversary.
```bash
python run_ccdo.py
```
Now you get a ccdo-protagonist and a ccdo-adversary.

### Evaluation
#### Datasets
If evaluating on sampled ones from the dataset, set the corresponding params in `configs/env/{env_name}.yaml`

If on the eval dataset, set:
```bash
eval_dataset: "val"    
dataset_state: "sample"
```
If on the test dataset, set:
```bash
eval_dataset: "test"    
dataset_state: "no_sample"
```


#### Eval trained RL agent

##### 1. Eval protagonist without adversary:

Firstly, set params `ckpt_path` to trained protagonist checkpoint path(ckpt) and its  dir to `evaluate_savedir` in `greedy_eval.yaml`. e.g.
```bash
evaluate_savedir: {root_dir}/logs/train/runs/{env.name}{env.num_loc}/am-{env.num}{env.num_loc}/{time} 
ckpt_path: {root_dir}/logs/train/runs/{env.name}{env.num_loc}/am-{env.num}{env.num_loc}/{time}/rl4co/xxxxxxx/checkpoints/xxx.ckpt

```
Then run the command:
```bash
python run.py evaluate=greedy_eval
```
You can find detailed results in the protagonist logdir.
> Tips:  
> If you evaluate on a sampled eval dataset,  you must run a CCDO algorithm and evaluate any protagonist(is or is not CCDO-RL) on it first. Except for `evaluate_savedir` and `ckpt_path`, also set `evaluate_psro_dir` to the ccdo alogrithm directory.
>

##### 2. Eval ccdo-protagonist with ccdo-adversary:

Set `evaluate_prog_dir` to ccdo logdir.
Set `eval_withadv` to true.
Run:
```bash
python run_ccdo.py evaluate=greedy_eval_ccdo
```
> Tips:  
> Must do this first if evaluating any with ccdo-adversary.

##### 3. Eval ccdo-protagonsit without adversary:
Following the last one, set  'eval_withadv` to false.
Still run:
```bash
python run_ccdo.py evaluate=greedy_eval_ccdo
```
##### 4. Eval protagonist with ccdo-adversary:

Modify `evaluate` in main_ccdo_frame.yaml as `eval_other_with_ccdo_adver.yaml` firstly.
In `eval_other_with_ccdo_adver.yaml`, set `evaluate_adv_dir` to the ccdo logdir path, and `eval_rl_prog` to `true`.
Set 'rl_prog_dir' and 'rl_prog_pth' to trained protagonist dir and ckpt.

Then run the command:
```bash
python run_ccdoadv_eval.py
```







#### Eval heuristic algorithms:
##### 1. Eval heuristic algorithm without adversary:


Firstly, set params `ckpt_path` to trained protagonist checkpoint path(ckpt) and its  dir to `evaluate_savedir` in `baseline_eval.yaml`. e.g.
Select the baseline method in `baseline`.
Then run the command:
```bash
python run.py evaluate=baseline_eval
```
##### 2. Eval heuristic-algorithm with ccdo-adversary:

Modify `evaluate` in main_ccdo_frame.yaml as `eval_other_with_ccdo_adver.yaml` firstly.
In `eval_other_with_ccdo_adver.yaml`, set `evaluate_adv_dir` to the ccdo logdir path, and `eval_baseline_prog` to `true`.
Set 'rl_prog_dir' and 'rl_prog_pth' to trained protagonist dir and ckpt
Select the baseline method in `baseline_heur`.
Then run the command:
```bash
python run_ccdoadv_eval.py
```
