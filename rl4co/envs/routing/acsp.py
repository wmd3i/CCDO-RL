from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length, get_padded_tour_length
from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.routing.env_utils import *
log = get_pylogger(__name__)


class ACSPEnv(RL4COEnvBase):
    """
    Covering Salesman Problem environment
    At each step, the agent chooses a city to visit. 
    The reward is 0 unless all the cities are visited or be covered by at least 1 city on the tour.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        num_loc: number of locations (cities) in the CSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "acsp"
    stoch_params = {
                        0: [0.6, 0.2, 0.2],
                        1: [0.8, 0.2, 0.0],
                        2: [0.8, 0.,  0.2],
                        3: [0.4, 0.3, 0.3]
                    }
    
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_cover: float = 0.,
        max_cover: float = 0.25,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_cover = min_cover
        self.max_cover = max_cover
        print(f" cover range {self.min_cover} - {self.max_cover}")
        self._make_spec(td_params)

        ACSPEnv.stoch_idx = kwargs.get("stoch_idx")

    @staticmethod
    def get_covered_guidence_vec(covered_node_bool, curr_distance):
        '''
        current covered node gi *= ci / N, ci is the order of the overlay points (from 1-N)
        and N is the total number of overlay points.
        covered_node_bool: [batch, num_loc], torch.bool
        curr_distance: curr node i, distance to all other nodes, [batch, num_loc], torch.bool
        
        '''
        curr_distance[~covered_node_bool] =  10000
        i, idx = curr_distance.sort(dim=-1)
        _, distance_sort_idx = idx.sort(dim=-1)
        covered_sort = torch.where(covered_node_bool, distance_sort_idx + 1,
                                   torch.zeros_like(covered_node_bool))
        curr_covered_num = covered_node_bool.sum(dim=-1)     # [batch, ]
        return covered_sort / (curr_covered_num.unsqueeze(-1) + 1e-5)
        
        
        
    def _step(self, td: TensorDict) -> TensorDict:
        '''
        update state:
            current node
        '''
        batch_size = td.batch_size[0]
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]       

        # get covered node of current node
        locs = td["locs"]       # [batch, num_loc,2]
        # curr_locs = locs.gather(-2, current_node[None, ..., None].repeat(1,1,2)).squeeze(0).unsqueeze(-2)     # [batch, 2]
        curr_locs = locs[range(batch_size), current_node][:, None, :].repeat(1, self.num_loc, 1)     # [batch, 2]
        curr_dist = (locs - curr_locs).norm(p=2, dim=-1)        #[batch, num_loc]

        # covered node: cover < min_cover, prob p to cover in (media_cover, max_cover): p=softmax(dij)
        covered_node = curr_dist < self.min_cover
        # in media distance, prob p to cover: p=softmax(dij)
        max_cover = td["stochastic_maxcover"][range(batch_size), current_node][..., None].repeat(1, self.num_loc)
        media_cover = (curr_dist > self.min_cover) & (curr_dist < max_cover)
        # tmp = torch.rand((batch_size, self.num_loc)).to(td.device)
        # masked = curr_dist * media_node
        # masked = torch.where(media_node, masked, -torch.ones_like(masked) * 1e5)
        # prob = torch.softmax((masked), -1)
        # media_cover = tmp <= prob
        covered_node[range(batch_size), current_node] = True
        covered_node[media_cover] = True

        td["covered_node"][covered_node] = True       #[batch, num_loc]
        curr_covered_guidence_vec = ACSPEnv.get_covered_guidence_vec(covered_node, curr_dist.clone())
        td["guidence_vec"] *= torch.where(covered_node, curr_covered_guidence_vec,
                                                        torch.ones_like(covered_node))
        
        # # Set not visited to 0 (i.e., we visited the node)
        # in csp, covered node still can be visited
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), False
        )

        #  done when all nodes are visited or covered
        done = (torch.sum(td["covered_node"], dim=-1) == self.num_loc) | (torch.sum(available, dim=-1) == 0)
        
        # # set complete solution: the first node's action_mask to False to avoid softmax(all -inf)=> nan error
        done_idx = torch.nonzero(done.squeeze())  #[batch]
        done_endnode = current_node[done_idx]   #[batch]
        # first set the done data's action_mask all to False
        available[done_idx, :] = False
        available[done_idx, done_endnode] = True
        
        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "guidence_vec": td["guidence_vec"],
                "covered_node": td["covered_node"],
                "reward": reward,
                "done": done,
            },
        )

        return td


    
    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if td is not None:
            init_locs = td["locs"]
            max_cover = td["max_cover"]
            stochastic_maxcover = td["stochastic_maxcover"]
            weather = td["weather"]
            min_cover = td["min_cover"]
        else:
            init_locs = None

        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.to(device)

        if init_locs is None:       # if no td, generate data 
            data = self.generate_data(batch_size=batch_size).to(device)
            init_locs = data["locs"]
            max_cover = data["max_cover"]
            stochastic_maxcover = data["stochastic_maxcover"]
            weather = data["weather"]
            min_cover = data["min_cover"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # start from depot
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        # get covered node of depot
        curr_locs = init_locs[range(batch_size[0]), current_node][:, None, :].repeat(1, num_loc, 1)     # [batch, 2]
        curr_dist = (init_locs - curr_locs).norm(p=2, dim=-1)        #[batch, num_loc]

        # covered node: cover < min_cover, prob p to cover in (media_cover, max_cover): p=softmax(dij)
        curr_covered_node = curr_dist < self.min_cover
        # in media distance, prob p to cover: p=softmax(dij)
        real_max_cover = stochastic_maxcover[range(*batch_size), current_node][..., None].repeat(1, self.num_loc)
        media_cover = (curr_dist >= self.min_cover) & (curr_dist < real_max_cover)
        curr_covered_node[media_cover] = True
        covered_node = torch.zeros(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means covered, i.e. but action is still allowed
        covered_node[curr_covered_node] = True       #[batch, num_loc]


        curr_covered_guidence_vec = ACSPEnv.get_covered_guidence_vec(covered_node, curr_dist.clone())

        guidence_vec = torch.ones(
            (*batch_size, num_loc), dtype=torch.float64, device=device
        )  # update while decoding, init 1. covered then decrease
        # 
        i = torch.ones((*batch_size, 1), dtype=torch.int64, device=device)
        guidence_vec *= torch.where(covered_node, curr_covered_guidence_vec,
                                                        torch.ones_like(covered_node))
        

        
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[:, 0] = False # can not visited depot

        

        

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "covered_node": covered_node,
                "guidence_vec": guidence_vec,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "min_cover": min_cover,
                "max_cover": max_cover,
                "stochastic_maxcover": stochastic_maxcover,
                "weather": weather
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.bool,
            ),
            covered_node=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.int64,
            ),
            guidence_vec=UnboundedContinuousTensorSpec(
                shape=(self.num_loc),
                dtype=torch.float64,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        # if self.check_solution:
        #     self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_padded_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are all visited or covered"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"
            
    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (        # batch, num_loc, 2
            torch.FloatTensor(*batch_size, self.num_loc, 2)
            .uniform_(self.min_loc, self.max_loc)
            ).to(self.device)
        
        max_cover = (       # batch, num_loc, 
            torch.FloatTensor(*batch_size, self.num_loc)
            .uniform_(self.min_cover, self.max_cover)
            ).to(self.device)
        
        # weather
        weather = (         # batch, 3
            torch.FloatTensor(*batch_size, 3)
            .uniform_(-1, 1)
        ).to(self.device)

        stochastic_maxcover = get_stoch_var(max_cover.to("cpu"),
                                                locs.to("cpu"), 
                                                weather[:, None, :].
                                                repeat(1, self.num_loc, 1).to("cpu"),
                                                None,
                                                stoch_idx=ACSPEnv.stoch_idx).squeeze(-1).float().to(self.device)

        stochastic_maxcover = torch.clamp(stochastic_maxcover, min=self.min_cover, max=self.max_cover)
        min_cover = torch.ones_like(max_cover) * self.min_cover
        return TensorDict({"locs": locs,
                           "min_cover": min_cover,
                           "max_cover": max_cover,
                           "stochastic_maxcover": stochastic_maxcover,
                           "weather": weather,
                           }, batch_size=batch_size)
    


    def reset_stochastic_var(self, td, adver_out):
        td = self.reset_stochastic_maxcover(td, adver_out)
        return td
    
    def reset_stochastic_maxcover(self, td, adver_action):
        '''
        adver_action: batch ,9
        '''
        
        locs_cust = td["locs"].clone()
        stochastic_maxcover = get_stoch_var(td["stochastic_maxcover"].to("cpu"),
                                                locs_cust.to("cpu"), 
                                                td["weather"][:, None, :].
                                                repeat(1, self.num_loc, 1).to("cpu"),
                                                adver_action[:, None, ...].to("cpu"),
                                                stoch_idx=ACSPEnv.stoch_idx).squeeze(-1).float().to(td.device)
        stochastic_maxcover = torch.clamp(stochastic_maxcover, min=self.min_cover, max=self.max_cover)
        # stochastic_maxcover = (stochastic_maxcover - stochastic_maxcover.min())* self.max_cover / (stochastic_maxcover.max() - stochastic_maxcover.min())
        td.set("stochastic_maxcover", stochastic_maxcover)
        return td
    

