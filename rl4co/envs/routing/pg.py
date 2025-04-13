from typing import Optional
from typing import Iterable, Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torch import Tensor
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.data.dataset import TensorDictDataset

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.pylogger import get_pylogger
import random
import numpy as np
from rl4co.envs.routing.env_utils import *
log = get_pylogger(__name__)


class PGEnv(RL4COEnvBase):
    """
    Adversarial time window Orienteering Problem environment
    At each step, the agent chooses a city to visit. The reward is -cost if attacked else 0. Finally add the undetected attack.
    Args:
        num_loc: number of locations (cities) in the OP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """
    name = "pg"
    
    
    def __init__(
        self,
        num_loc: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.prob_scale = 0.0   # attacker prob scale: [prob_scale, 1)
        PGEnv.stoch_idx = kwargs.get("stoch_idx")

    def dataset(self, batch_size=[], phase="train", filename=None, **kw):
        """Return a dataset of observations
            in pg, load data from presaved_data
        """
        
        if phase == "train":
            filename = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_val.npz"   # _test
        elif phase == "val":
            filename = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_val.npz"
        elif phase == "test":
            filename = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_test.npz"   # _test
        f = filename
        log.info(f"Loading {phase} dataset from {f}")
        try:
            if isinstance(f, Iterable) and not isinstance(f, str):
                names = getattr(self, f"{phase}_dataloader_names")
                return {
                    name: TensorDictDataset(self.load_data(_f, batch_size))
                    for name, _f in zip(names, f)
                }
            else:
                td = self.load_data(f, batch_size)
        except FileNotFoundError:
            log.error(
                f"Provided file name {f} not found. Make sure to provide a file in the right path first or "
                f"unset {phase}_file to generate data automatically instead"
            )
            td = self.generate_data(batch_size)

        if "sample_lst" in kw.keys():
            if isinstance(kw["sample_lst"], list):
                td = td[kw["sample_lst"], ...]
                print("size after sample: ", td["locs"].shape)
        return TensorDictDataset(td)
    

    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file
        Normalize data
        opswtw data: test and val data are the same 
        """
        if not batch_size:
            if "val" in fpath:
                batch_size = 10000
            elif "test" in fpath:
                batch_size = 100
        if isinstance(batch_size, list):
            batch_size = batch_size[0]
        td_load = load_npz_to_tensordict(fpath)[:batch_size, ...]
        locs = td_load["locs"]
        locs[..., 0] /= 200
        locs[..., 1] /= 50
        td_load.set("locs", locs)
        td_load.set("tw_low", td_load["tw_low"][:batch_size, ...] / 100)
        td_load.set("tw_high", td_load["tw_high"][:batch_size, ...] / 100)
        td_load.set("maxtime", td_load["maxtime"][:batch_size, ...] / 100)
        cost = td_load["prize"][:batch_size, ...]
        td_load.set("cost", cost)
        
        if ("val" in fpath):  # load other data for valand test
            part_pth = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_val_part_data.npz"
            td_load_part = load_npz_to_tensordict(part_pth)[:batch_size, ...]
            attack_prob = td_load_part["attack_prob"]
            real_prob = td_load_part["real_prob"]
            weather = td_load_part["weather"]
        elif "test" in fpath:
            part_pth = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_test_part_data.npz"
            td_load_part = load_npz_to_tensordict(part_pth)[:batch_size, ...]
            attack_prob = td_load_part["attack_prob"]
            real_prob = td_load_part["real_prob"]
            weather = td_load_part["weather"]
        else:
            weather = (
                torch.FloatTensor(batch_size, 3)
                .uniform_(-1, 1)
            ).to(self.device)
            # generate p: saved
            attack_prob = (
                torch.FloatTensor(batch_size, self.num_loc)
                .uniform_(0, 1)
            ).to(self.device)
            attack_prob[:, 0] = 0.
        
            # influence prize by attack prob
            # tmp = torch.rand((batch_size, self.num_loc)).to(self.device)
            real_prob = get_stoch_var(attack_prob.to("cpu"),
                                     locs.to("cpu"),
                                     weather[:, None, :].
                                     repeat(1, self.num_loc, 1).to("cpu"),
                                     None,
                                     stoch_idx=PGEnv.stoch_idx).squeeze(-1).float().to(self.device)
            real_prob = torch.clamp(real_prob, min=0., max=1.)
        
        real_prob = real_prob * (1 - self.prob_scale) + self.prob_scale     # scale uniform(0,1) to [prob-scale, 1)
        real_prob[:, 0]=0
        attacked = real_prob > attack_prob
        stoch_cost = cost.clone() * attacked
        

        td_load.set("attack_prob", attack_prob)
        td_load.set("real_prob", real_prob)
        td_load.set("stochastic_cost", stoch_cost)
        td_load.set("weather", weather)

        return td_load
    
    def generate_data(self, batch_size) -> TensorDict:
        # load the training data of pg
        data_pth = "/home/yuheng/rl4co/data"+str(PGEnv.stoch_idx)+"/pg/pg"+str(self.num_loc)+"_train.npz"

        x_dict = self.load_data(data_pth, batch_size)
        whole_size = x_dict.batch_size[0]

        if isinstance(batch_size, list):
            batch_size = batch_size[0]
        assert whole_size >= batch_size, "Required size exceeds data size"

        locs = x_dict["locs"][:batch_size, ...]
        tw_low = x_dict["tw_low"][:batch_size, ...]
        tw_high = x_dict["tw_high"][:batch_size, ...]
        cost = x_dict["cost"][:batch_size, ...]
        maxtime = x_dict["maxtime"][:batch_size, ...]
        attack_prob = x_dict["attack_prob"][:batch_size, ...]
        stochastic_cost = x_dict["stochastic_cost"][:batch_size, ...]
        real_prob = x_dict["real_prob"][:batch_size, ...]
        weather = x_dict["weather"][:batch_size, ...]
        
        return TensorDict(
            {
            "locs": locs,
            "tw_low": tw_low,
            "tw_high": tw_high,
            "cost": cost,
            "stochastic_cost": stochastic_cost,
            "attack_prob": attack_prob,
            "real_prob": real_prob,
            "weather": weather,
            "maxtime": maxtime,
        },
        batch_size=batch_size,
        device=self.device
        )
    
    
    
    
    @staticmethod
    def get_adjacent_matrix(x: Tensor, y: Tensor):
        return (x - y).norm(p=2, dim=-1)

    def _reset(self, 
               td: Optional[TensorDict] = None,
               batch_size: Optional[list] = None,
               ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size


        if td is None:
            td = self.generate_data(batch_size)
        
        locs = td["locs"]
        curr_time = torch.zeros((*batch_size, ), device=self.device)    # batch,
        curr_cost = torch.zeros((*batch_size,), device=self.device)
        curr_node = torch.zeros((*batch_size, ), device=self.device, dtype=torch.int64)
        row_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1)
        col_locs = locs[..., None, :].repeat(1, 1, self.num_loc, 1).transpose(1, 2)
        adj = PGEnv.get_adjacent_matrix(row_locs, col_locs)     # batch, size, size
        penalty = torch.zeros((*batch_size, ), device=self.device)
        # begin from the depot
        action_mask = torch.ones((*batch_size, self.num_loc), device=self.device).bool()    # True if visted
        action_mask[:, 0] = False       # mask depot
        action_mask[td["tw_low"] > td["maxtime"]] = False

        visited = torch.zeros((*batch_size, self.num_loc), dtype=torch.uint8, device=self.device)
        visited[:, 0] = 1
        done = torch.zeros((*batch_size,), device=self.device, dtype=torch.bool)

        td_reset = TensorDict(
            {
                "locs": locs,
                "tw_low": td["tw_low"],
                "tw_high": td["tw_high"],
                "cost": td["cost"],
                "curr_cost": curr_cost,
                "stochastic_cost": td["stochastic_cost"],
                "attack_prob": td["attack_prob"],
                "real_prob": td["real_prob"],
                "weather": td["weather"],
                "maxtime": td["maxtime"],
                "tour_time": curr_time,
                "adj": adj,
                "penalty": penalty,
                "action_mask": action_mask,
                "current_node": curr_node,
                "visited": visited,
                "done": done
            },
            batch_size=batch_size,

        )
        return td_reset
    
    def _step(self, td: TensorDict) -> TensorDict:
        '''
        update state:   
            current tour time: wait util low time window 
            current node(and visited node),
            done: if exceed maxtime, or select depot, or cannot back to depot with enough time
            cost: substract cost for defended attack, except exceed high time thres.
        constrains: 
            penalty if exceeds maxtime, and done =True
        '''
        last_node = td["current_node"]
        current_node = td["action"]     # 1-size， 0: index of the depot
        batch_size = last_node.shape[0]
        # update visited node
        visited = td["visited"]
        visited[range(batch_size), current_node] = 1
        # update tour time 
        tour_time = td["tour_time"]
        adj = td["adj"]
        time = adj[range(batch_size), last_node, current_node]
        tour_time += time

        
        curr_tw_low = td["tw_low"][range(batch_size), current_node]
        curr_tw_high = td["tw_high"][range(batch_size), current_node]
        # add wait time if not in time window's low thres
        wait_idx = tour_time < curr_tw_low      # batch,  bool
        waited_time = curr_tw_low - tour_time
        tour_time += waited_time * wait_idx
        # penalty if exceeds maxtime
        penalty = td["penalty"]
        penalty_solution = tour_time > td["maxtime"][:, 0]
        penalty -= (self.num_loc * penalty_solution) * (~td["done"].squeeze(-1))
        done = penalty_solution[..., None]     # if exceed maxtime, solution done
        # if curr node = 0, and this is second time to depot, done=True
        back_to_depot = (current_node==0) & (visited[:, 1:].int().sum(-1) > 0)
        back_to_depot_idx = torch.nonzero(back_to_depot)
        done[back_to_depot_idx] = True
        back_to_depot = self.check_if_back(td, current_node, tour_time)
        done[back_to_depot] = True
        # add cost to reward(maximize) if defend 
        curr_cost = td["curr_cost"]
        # multiply realprob
        real_cost = td["stochastic_cost"][range(batch_size), current_node]
        real_prob_curr = td["real_prob"][range(batch_size), current_node]
        # still penalty attack's cost if exceeds the high thres
        curr_cost = td["curr_cost"]
        curr_penalty = tour_time >= curr_tw_high
        real_cost[curr_penalty] = 0
        curr_cost += real_cost*real_prob_curr



        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "tour_time": tour_time,
                "penalty": penalty,
                "curr_cost": curr_cost,
                "done": done,

            }
        )
        td.set("action_mask", self.get_action_mask(td))
        
        return td
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # mask_loc=False means this node cannot be chosen

        # Nodes that cannot be visited are already visited
        mask_loc = ~td["visited"].to(torch.bool)
        done_idx = torch.nonzero(td["done"].squeeze(-1))
        done_node_mask = torch.zeros((self.num_loc)).to(torch.bool)
        mask_loc[done_idx, :] = done_node_mask.to(mask_loc.device)

        tourtime_exceed_hightw = td["tour_time"][..., None].repeat(1, self.num_loc) >  td["tw_high"]
        mask_loc[tourtime_exceed_hightw] = False
        mask_loc[range(td["visited"].shape[0]), 0] = True    
        
        return mask_loc

    def check_if_back(self, td, current_node, tour_time):
        adj = td["adj"]
        curr_node = current_node
        distance_back_to_depot = adj[range(td["visited"].shape[0]), curr_node, 0]
        tour_time = tour_time
        back_to_depot = (tour_time + distance_back_to_depot) > td["maxtime"][:, 0]
        return back_to_depot

    def get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        return self.get_final_reward(td, actions)

    def get_final_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        '''
        collect undefended attack's costs
        '''
        reward = td["curr_cost"] + td["penalty"]

        return reward

    
    
    def reset_stochastic_prob(self, td, adver_action):
        '''
        adver_action: batch ,9
        '''
        batch_size = td["cost"].size(0)
        locs_cust = td["locs"].clone()
        real_prob = get_stoch_var(td["attack_prob"].to("cpu"),
                                                locs_cust.to("cpu"), 
                                                td["weather"][:, None, :].
                                                repeat(1, self.num_loc, 1).to("cpu"),
                                                adver_action[:, None, ...].to("cpu"),
                                                stoch_idx=PGEnv.stoch_idx).squeeze(-1).float().to(td.device)
        real_prob = torch.clamp(real_prob, min=0., max=1.)
        real_prob[:, 0] = 0                         
        td.set("real_prob", real_prob)
        # reset cost by new attack prob
        attack_prob = td["attack_prob"]
        attacked = real_prob > attack_prob
        cost = td["cost"]
        stoch_cost = cost.clone() * attacked
        td.set("stochastic_cost", stoch_cost)
        return td
    
    def reset_stochastic_var(self, td, adver_out):
        td = self.reset_stochastic_prob(td, adver_out)
        return td

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, **kwargs):
        ''''
        print the route chosen by agent
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm, colormaps

        if actions.device != "cpu":
            actions = actions.to("cpu")
        num_routine = (actions == 0).sum().item() + 2
        base = colormaps["nipy_spectral"]
        color_list = base(np.linspace(0, 1, num_routine))
        cmap_name = base.name + str(num_routine)
        out = base.from_list(cmap_name, color_list, num_routine)

        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        td = td.detach().cpu()

        if actions is None:
            actions = td.get("action", None)
        
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]
        
        locs = td["locs"]
        prize = td["prize"]
        # real_node_prize = td["real_node_prize"]
        real_prize = td["stochastic_prize"]

        # add the depot at the first action 
        actions = torch.cat([torch.tensor([0]), actions])

        # gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            locs = locs

        # Cat the first node to the end to complete the tour
        x, y = locs[:, 0], locs[:, 1]

        # plot depot
        ax.scatter(
            locs[0, 0],
            locs[0, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=2,
            marker="s",
            alpha=1,
        )

        # plot visited nodes
        ax.scatter(
            x[1:],
            y[1:],
            edgecolors=cm.Set2(0),
            facecolors="none",
            s=50,
            linewidths=2,
            marker="o",
            alpha=1,
        )

        move_x = 0.08        # text整体位置向右平移
        # plot prize bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.01,   # width
                    2*prize[node_idx]/10,       # ?
                    edgecolor=cm.Set2(0),
                    facecolor=cm.Set2(0),
                    fill=True,
                )
            )
            
        # text node idx
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] - 0.18 + move_x,
                locs[node_idx, 1] - 0.025,
                f"{node_idx}:",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color="b",
            )

        # text prize
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] + move_x,
                locs[node_idx, 1] - 0.025,
                f"{prize[node_idx].item():.2f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(0),
            )
        
        # plot real prize bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.005,
                    2*real_prize[node_idx]/10,      # ?
                    edgecolor=cm.Set2(1),
                    facecolor=cm.Set2(1),
                    fill=True,
                )
            )

        # text real prize
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] - 0.085+ move_x,
                locs[node_idx, 1] - 0.025,
                f"{real_prize[node_idx].item():.2f} |",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(1),
            )

        # text depot
        ax.text(
            locs[0, 0],
            locs[0, 1] - 0.025,
            "Depot",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(2),
        )

        # plot actions
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx + 1]]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color=out(color_idx),
                lw=1,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
                size=15,
                annotation_clip=False,
            )
        
        # Setup limits and show
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
        if kwargs["save_pt"]:
            plt.savefig(kwargs["save_pt"])