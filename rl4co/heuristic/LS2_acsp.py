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

import time
import random
from .LS_2opt_acsp import LS_acsp_2opt
import math
class LocalSearch2_acsp:
    def __init__(self, td) -> None:
        super().__init__()
        '''
         improve the cost of a solution by either deleting a node on the tour
        if the resulting solution is feasible or by extracting a node and substituting it with a promising sequence of nodes.

        td: TensorDict, after call .reset() function
        td["locs"]: [batch, num_customer, 2], customers and depot(0)
        td["min_cover"]: [batch, num_customer]
        td["max_cover"]: [batch, num_customer], expected max cover, heter
        td["stochastic_maxcover"] : [batch, num_customer],  real max cover
        td["weather"]: [batch, 3], 

        td["action_mask"]: [batch, num_customer], 0 is not allowed i.e. False; others True
        td["reward"]:  [batch, ], 0
        others
        
        forward(): call this to get all solutions, a nested list, call convert_to_fit_npz as for savez
        '''
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")  # #
        self.td = td
        self.batch_size = self.td["locs"].shape[0]
        self.num_loc = self.td["locs"].shape[1]
        self.locs = td["locs"]
        self.min_cover = td["min_cover"]
        self.max_cover = td["max_cover"]
        self.stoch_max_cover = td["stochastic_maxcover"]

        self.instance_idx = 0
        self.ls_2opt = LS_acsp_2opt(self.locs[self.instance_idx])

        self.max_iters = 25
        self.J=100
        self.T=10
        self.K=10
        self.search_iters = 30
        self.print_if = False
    def init_solution(self):
        '''
        begin from the depot
        '''
        ids = torch.arange(1, self.num_loc)
        uncovered_indices = ids
        num_uncovered = self.num_loc - 1
        tour = [0]

        # adding the node until all nodes are covered
        while num_uncovered>0:
            n_selected = random.choice(uncovered_indices)
            tour.append(int(n_selected))
            cover_indices = self.cover_idx(int(n_selected))
            uncovered_indices = torch.tensor([idx for idx in uncovered_indices if idx not in cover_indices])
            num_uncovered = uncovered_indices.shape[0]
        
        return torch.tensor(tour)
    
    def dist(self, tour):
        loc = self.locs[self.instance_idx, tour]
        # 计算路径长，包括回到depot的路径长
        return torch.sqrt(torch.square(loc[1:]-loc[:-1]).sum(-1)).sum() + torch.sqrt(torch.square(loc[0]-loc[-1]).sum())
    
    def check_cover(self, tour):
        "check if all nodes are covered"
        mask = torch.zeros(self.num_loc)
        for idx in tour:
            cover_indices = self.cover_idx(idx)
            mask[cover_indices]=1
        return mask.all()
    
    def cover_idx(self, chosen_idx):
        "return the index of covered nodes"
        dists = torch.sqrt(torch.square(self.locs[self.instance_idx]-self.locs[self.instance_idx, chosen_idx]).sum(-1))     # [num_loc,]
        radius = self.max_cover[self.instance_idx, chosen_idx]

        return dists.argsort()[torch.sort(dists)[0] <= radius]
    
    def subsitute_by_neighbor(self, del_tour, del_pos, del_node, ori_cost):
        '''
        Replace with the nearest neighbor node, and the cost path is shorter, if not return None
        '''
        neighbours = self.cover_idx(del_node)[1:]
        neighbours = [i for i in neighbours if i not in del_tour]
        neighbours = neighbours[:min(self.T, len(neighbours))]
        # print(del_node, del_tour, neighbours)
        best_cost = torch.inf
        best_tour = None
        for node in neighbours:
            # tour = np.insert(del_tour, del_pos, node)
            tour_insert = del_tour.clone().tolist()
            tour_insert.insert(del_pos, node)
            tour_insert = torch.tensor(tour_insert, dtype=torch.int32)
            cost_now = self.dist(tour_insert)
            if cost_now<ori_cost:
                if self.check_cover(tour_insert):
                    if cost_now < best_cost:
                        best_cost = cost_now
                        best_tour = tour_insert

        return best_tour

    def improve_process(self, tour):
        '''
        Traverse all nodes: Delete one, if still feasible, improve=True;
        Otherwise, replace it with some other nodes
        '''
        improve = False
        ori_cost = self.dist(tour)
        Ns = tour.shape[0]
        del_pos = 1     # depot不能删除
        while del_pos < Ns:
            del_node = tour[del_pos]
            # del_tour = np.delete(tour, del_pos)
            del_tour = torch.concat((tour[:del_pos], tour[del_pos+1:]), dim=0)
            if self.check_cover(del_tour):
                improve = True
                tour = del_tour
            else:
                subsituted_tour = self.subsitute_by_neighbor(del_tour, del_pos, del_node, ori_cost)
                if subsituted_tour is not None:
                    tour = subsituted_tour
                    improve = True
                del_pos = del_pos + 1
            Ns = tour.shape[0]

        return improve, tour

    def perturbation_process(self, tour, ids):
        """
        """
        for i in range(self.K):
            # nodes_not_selected = np.setdiff1d(ids, tour)
            nodes_not_selected = torch.tensor([idx for idx in ids if idx not in tour])
            if nodes_not_selected.shape[0] == 0:
                break
            # node = np.random.choice(nodes_not_selected)
            probs = torch.ones(1)
            samples = torch.multinomial(probs, num_samples=1, replacement=False).to("cpu")
            node = nodes_not_selected[samples]
            best_pos, _ = self.best_add_position(tour, node)
            # tour = np.insert(tour, best_pos, node)
            tour_insert = tour.clone().tolist()
            tour_insert.insert(best_pos, node)
            tour = torch.tensor(tour_insert, dtype=torch.int32)
        return tour

    def best_add_position(self, tour, node):
        """
        From all the positions where a node can be inserted, find the best position:
        the path length increases the least after insertion
        """
        dist_ori = self.dist(tour)
        # Look up all insert position, calculate the increase of distance
        best_pos = 0
        minmum_increase = torch.inf
        for pos in range(1, tour.shape[0]):     # 不能插入在depot处
            tour_insert = tour.clone().tolist()
            tour_insert.insert(pos, node)
            tour_insert = torch.tensor(tour_insert, dtype=torch.int32)
            dist_increase = self.dist(tour_insert) - dist_ori
            if dist_increase<minmum_increase:
                best_pos = pos
                minmum_increase = dist_increase
        return best_pos, minmum_increase
    
    def gather_by_index(self, idx, dim=1, squeeze=True):
        """Gather elements from src by index idx along specified dim

        Example:
        >>> src: shape [permutes, 20, 2]
        >>> idx: shape [permutes, 3)] # 3 is the number of idxs on dim 0
        >>> Returns: [permutes, 3, 2]  # get the 3 elements from src at idx
        """
        src = self.locs[self.instance_idx][None, ...].repeat(idx.shape[0], 1, 1)
        expanded_shape = list(src.shape)
        expanded_shape[dim] = -1
        idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
        return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)
    
    def get_tour_length(self, ordered_locs):
        """Compute the total tour distance for a batch of ordered tours.
        Computes the L2 norm between each pair of consecutive nodes in the tour and sums them up.

        Args:
            ordered_locs: Tensor of shape [permu_nums, num_nodes, 2] containing the ordered locations of the tour
        """
        if ordered_locs.dim() == 2:
            ordered_locs = ordered_locs[None, ...]
        ordered_locs_next = torch.roll(ordered_locs, -1, dims=-2)
        ordered_locs_next[:, -1, :] = ordered_locs[:, -1, :]
        return self.get_distance(ordered_locs_next, ordered_locs).sum(-1)

    def get_distance(self, x, y):
        """Euclidean distance between two tensors of shape `[..., n, dim]`"""
        return (x - y).norm(p=2, dim=-1)
    
    def check_stoch_cover(self, tour):
        "real maxcover"
        mask = torch.zeros(self.num_loc)
        for idx in tour:
            cover_indices = self.stoch_cover_idx(idx)
            mask[cover_indices]=1
        return mask.all(), mask
    
    def stoch_cover_idx(self, chosen_idx):
        "return the index of real covered nodes"
        dists = torch.sqrt(torch.square(self.locs[self.instance_idx]-self.locs[self.instance_idx, chosen_idx]).sum(-1))     # [num_loc,]
        radius = self.stoch_max_cover[self.instance_idx, chosen_idx]

        return dists.argsort()[torch.sort(dists)[0] <= radius]
    
    # @profile(stream=open('log_mem_csp_LS2_perm.log', 'w+'))
    def add_uncovered_nodes(self, end_of_tour, uncovered):
        '''
            Add the uncovered node along the shortest path first,
            Update uncovered after adding one, and stop adding nodes if all nodes are covered
        '''

        best_tour = None
        best_length = 1e5
        uncovered = uncovered.squeeze().tolist()
        if not isinstance(uncovered, list):
            uncovered = [uncovered]
        times = min(self.search_iters, math.factorial(len(uncovered)))
        for i in range(times):
            perm = uncovered
            random.shuffle(perm)
            perm = torch.tensor(perm)
            tour = torch.concat((end_of_tour.unsqueeze(0), perm), dim=0)
            locs_permutes = self.gather_by_index(tour)
            dis = self.get_tour_length(locs_permutes)
            if dis < best_length:
                best_length = dis
                best_tour = tour
            
        return best_tour
    
    def forward(self):
        rewards = []
        routes = []
        st = time.time()
        for i in range(self.batch_size):
            print(f"search for data {i}")
            cost, solution = self.forward_single()
            # print(solution)
            self.instance_idx += 1
            rewards.append(cost)
            routes.append(solution)
        
        est = time.time()
        print(f"total time: {est - st}, mean time: {(est - st) / self.batch_size}")

        routes = convert_to_fit_npz(routes)

        mean_ = sum(rewards) / len(rewards)
        squared_deviations = [(value - mean_) ** 2 for value in rewards]
        var_ = sum(squared_deviations) / len(rewards)


        return {
            "routes": routes,
            "rewards": rewards,
            "mean reward": sum(rewards) / len(rewards),
            "var reward": var_,
            "time": est-st
        }
    # @profile(stream=open('log_mem_csp_LS2.log', 'w+'))
    def forward_single(self, tour=None, stop_cost = -1e6):

        if tour is None:
            tour = self.init_solution()
            # print("Initial Solution!!")

        ids = torch.arange(0, self.num_loc)

        best_cost = self.dist(tour)
        best_tour = tour.to(self.device)
        iter_no_change_outer = 0

        for i in range(self.max_iters):
            bestimprove = False
            iter_no_change = 0

            for j in range(self.J):
                improve = True
                while improve is True:
                    improve, tour = self.improve_process(tour)
                tour = self.ls_2opt.LS_2opt(tour, 100)
                cost_now = self.dist(tour)
                if cost_now < best_cost:
                    best_tour = tour
                    best_cost = cost_now
                    bestimprove = True
                    iter_no_change = 0
                else:
                    tour = best_tour
                    iter_no_change = iter_no_change + 1
                tour = self.perturbation_process(tour, ids)
                if iter_no_change > 50:
                    break
            if bestimprove is True:
                iter_no_change_outer = 0
                best_tour = self.ls_2opt.LS_2opt(best_tour, 200)
                cost_now = self.dist(best_tour)
                tour = best_tour
                best_cost = cost_now
            iter_no_change_outer = iter_no_change_outer + 1
            if iter_no_change_outer > 3:
                break
            if self.print_if:
                print('Iter,', i, ' cost:', best_cost)
            if best_cost <= stop_cost:
                return -1e-6, best_tour

        # 随机max cover检查是否都覆盖，没覆盖add到最后
        all_mask, mask = self.check_stoch_cover(best_tour)
        if not all_mask:
            uncovered_nodes = torch.nonzero(mask-1)
            best_uncovered_tour = self.add_uncovered_nodes(best_tour[-1], uncovered_nodes)

            # 逐个加入要加入的node，都覆盖就停止
            for add_node in best_uncovered_tour[1:]:
                best_tour = torch.concat((best_tour, add_node[None, ...]), dim=0)
                best_cost = self.dist(best_tour)
                if self.check_stoch_cover(best_tour):
                    break
        
        return best_cost, best_tour