import torch
import pandas as pd
from rl4co.envs import ACVRPEnv
from torch.nn.utils.rnn import pad_sequence
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from rl4co.utils.heuristic_utils import convert_to_fit_npz
import random

class Random_acvrp:
    def __init__(self, td) -> None:
        super().__init__()
        '''
        td: TensorDict
        td["locs"]: [batch, num_customer+1, 2], customers and depot(0)
        td["demand"]:  [batch, num_customer], expected demand
        td["real_demand"]:  [batch, num_customer]
        td["used_capacity"]: [batch, 1]
        td["vehicle_capacity"]: [batch, 1]
        
        td["visited"]: [batch, 1, num_customer+1]
        td["current_node"]: [batch, 1]
        td["weather"]: [batch, 3]
        
        forward(): call this to get all solutions, a nested list, call convert_to_fit_npz as for savez
        '''
        self.td = td
        self.num_customer = 0
        self.data_num = None      # [batch, num_customer, 2], distance to depot, demand
        self.routes = list()
        self.capacity = 1        # define by env
        
        
 

    def _get_reward(self, td: TensorDict, batch_actions):
        '''
        batch_actions: from 1 to num_customers
        '''
        batch_actions = torch.tensor(batch_actions, device=td["demand"].device, 
                                                   dtype=torch.int64) 
        # get the solution's penaltied idx
        loc_idx_penaltied = self.get_penalty_loc_idx(td, batch_actions)   #[batch, penalty_number]

        # Gather dataset in order of tour
        depot = td["locs"][..., 0:1, :]
        depot_batch = depot.repeat(1, loc_idx_penaltied.size(1), 1) # [batch,  penalty_number, 2]
        
        # get penaltied lcoations
        locs_penalty = td["locs"][..., 1:, :]* loc_idx_penaltied[..., None]       #[batch, penalty_number, 2]
        # get 0 pad mask
        posit = loc_idx_penaltied > 0  #[batch, penalty number]
        posit = posit[:,:, None]    #[batch, penalty number, 1]
        posit = posit.repeat(1, 1, 2)   #[batch, penalty number, 2]
        locs_penalty = torch.where(posit, locs_penalty, depot_batch)
        
        locs_ordered = torch.cat([depot, gather_by_index(td["locs"], batch_actions)], dim=1)
        cost_orig = -get_tour_length(locs_ordered)
        cost_penalty = -get_distance(depot_batch, locs_penalty).sum(-1) * 2
        # print(f"orig is {cost_orig}, penalty is {cost_penalty}")
        return cost_penalty + cost_orig
    
    @staticmethod
    def get_penalty_loc_idx(td: TensorDict, batch_actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot.
        if capacity is not exceeded, record the loc idx
        return penaltied location idx, [batch, penalty_number]
        """
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td["demand"].size()
        sorted_pi = batch_actions.data.sort(1)[0]
        
        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"


        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        real_demand_with_depot = torch.cat((-td["vehicle_capacity"], td["real_demand"]), 1)     # [batch, num_cust+1]
        d = real_demand_with_depot.gather(1, batch_actions)     # shape is same as batch_actions 

        used_cap = torch.zeros((td["demand"].size(0), 1), device=td["demand"].device)
        penaltied_idx = torch.zeros_like(td["demand"], device=td["demand"].device)      # [batch, num_customer]
        for i in range(batch_actions.size(1)):
            used_cap[:, 0] += d[
                    :, i
                ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            exceed_cap_bool = used_cap[:, 0] > td["vehicle_capacity"][:,0] + 1e-5        # 1 dim
            if any(exceed_cap_bool):
                # print("Used more than capacity")
                exceed_idx = torch.nonzero(exceed_cap_bool)     # [exceed_data_idx, 1]
                penaltied_node = batch_actions[exceed_idx, i] - 1     # [exceed_data_idx, 1], in"actions",customer node start from 1, substract 1 when be idx
                penaltied_idx[exceed_idx, penaltied_node] = 1        # set exceed idx to 1
                used_cap[exceed_idx, 0] = d[exceed_idx, i]
        # print(f"time of one loop: {end_3 - start_3}")
        return penaltied_idx
    
    
    def forward(self):
        
        self.num_customer = self.td["demand"].size(1)
        self.data_num = self.td["demand"].size(0)
        
        self.routes = [[i for i in range(1, self.num_customer+1)] for _ in range(self.data_num)]
        for sub in self.routes:
            random.shuffle(sub)
        # for i in range(self.data_num):
        #     # print(f"instance {i}")
        #     single_routes = self.forward_single(i)
        #     self.routes.append(single_routes)
        
        
        rewards = self._get_reward(self.td, self.routes)     #[batch]
        mean_reward = rewards.mean()
        # print('------Random for svrp-----')
        # print(f'Routes found are:{self.routes}, rewards are {rewards}, mean reward is {mean_reward} ')
        
        routes = convert_to_fit_npz(self.routes)
        return {"routes":routes,
                "rewards": rewards,
                "mean reward": mean_reward
            }  

    def forward_single(self, i):
        '''
        i: i th data, start from 0
        '''
        

        self.routes_single = list(range(1, self.num_customer+1))
        random.shuffle(self.routes_single)
        
            
        return self.routes_single
 