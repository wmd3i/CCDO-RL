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
import math
import time
import random
from .LS_2opt_acsp import LS_acsp_2opt
class LocalSearch1_acsp:
    def __init__(self, td) -> None:
        super().__init__()
        '''
        find improvements in a solution S by replacing some nodes of the current tour.
         achieves this in a two-step manner.
            First, LS1 deletes a fixed number of nodes.
            2. attempts to make the solution feasible by inserting new nodes into S
            
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
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.td = td
        self.batch_size = self.td["locs"].shape[0]
        self.num_loc = self.td["locs"].shape[1]
        self.locs = td["locs"]
        self.min_cover = td["min_cover"]
        self.max_cover = td["max_cover"]
        self.stoch_max_cover = td["stochastic_maxcover"]

        self.instance_idx = 0
        self.ls_2opt = LS_acsp_2opt(self.locs[self.instance_idx])


        self.max_iters = 500
        self.del_perc = 0.3
        self.mutate_iters = 15
        self.if_diverse = True
        self.if_mutate = True
        self.a = 0.1
        self.search_iters = 30
        self.print_enable = False
    

    def init_solution(self):
        '''
        从depot出发
        '''
        ids = torch.arange(1, self.num_loc)
        uncovered_indices = ids
        num_uncovered = self.num_loc - 1
        tour = [0]

        # 不断加入，直到所有节点都被覆盖
        while num_uncovered>0:
            n_selected = random.choice(uncovered_indices)
            tour.append(int(n_selected))
            cover_indices = self.cover_idx(int(n_selected))     # 在覆盖范围内的nodes的索引
            uncovered_indices = torch.tensor([idx for idx in uncovered_indices if idx not in cover_indices])
            num_uncovered = uncovered_indices.shape[0]
        
        return torch.tensor(tour)
    
    def check_stoch_cover(self, tour):
        "真实的maxcover，是否所有节点都被覆盖"
        mask = torch.zeros(self.num_loc)
        for idx in tour:
            cover_indices = self.stoch_cover_idx(idx)
            mask[cover_indices]=1
        return mask.all(), mask
    
    def stoch_cover_idx(self, chosen_idx):
        "返回在chosen_idx真实的覆盖范围内的idx"
        dists = torch.sqrt(torch.square(self.locs[self.instance_idx]-self.locs[self.instance_idx, chosen_idx]).sum(-1))     # [num_loc,]
        radius = self.stoch_max_cover[self.instance_idx, chosen_idx]

        return dists.argsort()[torch.sort(dists)[0] <= radius]
    
    def check_cover(self, tour):
        "是否所有节点都被覆盖"
        mask = torch.zeros(self.num_loc)
        for idx in tour:
            cover_indices = self.cover_idx(idx)
            mask[cover_indices]=1
        return mask.all()
    
    
    def cover_idx(self, chosen_idx):
        "返回在chosen_idx覆盖范围内的idx"
        dists = torch.sqrt(torch.square(self.locs[self.instance_idx]-self.locs[self.instance_idx, chosen_idx]).sum(-1))     # [num_loc,]
        radius = self.max_cover[self.instance_idx, chosen_idx]

        return dists.argsort()[torch.sort(dists)[0] <= radius]

    def dist(self, tour):
        loc = self.locs[self.instance_idx, tour]
        # 计算路径长，包括回到depot的路径长
        return torch.sqrt(torch.square(loc[1:]-loc[:-1]).sum(-1)).sum() + torch.sqrt(torch.square(loc[0]-loc[-1]).sum())
    
    def delete_nodes_probs(self, tour):
        "返回和tour大小一样的,对应每个节点删除的概率"
        tour = tour.clone()
        dist_ori = self.dist(tour)
        Cs=[1e-6]      # depot的删除概率无限小
        for idx in tour[1:]:        # 不能删除depot
            tour_del = tour[tour!=idx]      # 去掉idx的剩下的节点
            del_dist = self.dist(tour_del)
            dist_improve = max((dist_ori - del_dist),0)
            Cs.append(dist_improve)
        Cs = torch.tensor(Cs).to(self.device)/sum(Cs)
        return Cs

    

    def feasible_process(self, tour, nodes_not_select):
        '''
        "不断加入节点直到所有节点都被覆盖"
        这里一开始的nodes_not_select不包括被删掉的ndoes，可能出现没有节点能覆盖这些del nodes的情况
        '''
        while not self.check_cover(tour):
            scores, best_poss = self.add_node_score(tour, nodes_not_select)
            if (scores > 1e5).all() and best_poss.sum() == 0:   # 当没有点能覆盖del nodes，添加进去被删除的nodes
                ids = torch.arange(0, self.num_loc)
                nodes_not_select = torch.tensor([idx for idx in ids if idx not in tour]) 
                scores, best_poss = self.add_node_score(tour, nodes_not_select)
            # if scores==[]:
            #     break
            add_node = nodes_not_select[torch.argmin(scores)]      # 得分最低，最好
            add_pos = best_poss[torch.argmin(scores)]
            tour = tour.tolist()
            tour.insert(add_pos, add_node)
            tour = torch.tensor(tour)
            # nodes_not_select = torch.setdiff1d(nodes_not_select, add_node)
            nodes_not_select = torch.tensor([idx for idx in nodes_not_select if idx not in add_node])
        return tour

    def add_node_score(self, tour, nodes_not_select):
        "返回所有可加入的节点加入tour的得分，以及最佳位置"
        uncovered_ids = self.get_uncovered_ids(tour)
        scores=[]
        best_poss=[]
        for node in nodes_not_select:
            # if add this node. How much uncovered nodes can be covered by this node
            covered_ids = self.cover_idx(node)
            # nums_thisnode_can_cover = torch.intersect1d(covered_ids, uncovered_ids).shape[0]       # 该节点对未覆盖节点集中的覆盖大小
            nums_thisnode_can_cover = len([idx for idx in covered_ids.tolist() if idx in uncovered_ids.tolist()])
            if nums_thisnode_can_cover > 0: # 
                # find the best position to insert this node
                best_pos, minmum_increase = self.best_add_position(tour, node)
                score = minmum_increase/nums_thisnode_can_cover**2
            else:
                score = torch.inf
                best_pos = 0
            scores.append(score)
            best_poss.append(best_pos)
        return torch.tensor(scores), torch.tensor(best_poss)

    def best_add_position(self, tour, node):
        "node可插入的所有位置, 找到最佳位置：插入后路径总长增加最少的"
        dist_ori = self.dist(tour)
        # Look up all insert position, calculate the increase of distance
        best_pos = 0
        minmum_increase = torch.inf
        for pos in range(1, tour.shape[0]):     # 不能插入在depot处
            tour_insert = tour.clone().tolist()
            tour_insert.insert(pos, node)
            tour_insert = torch.tensor(tour_insert)
            dist_increase = self.dist(tour_insert) - dist_ori
            if dist_increase<minmum_increase:
                best_pos = pos
                minmum_increase = dist_increase
        return best_pos, minmum_increase

    def get_uncovered_ids(self, tour):
        mask = torch.zeros(self.num_loc)
        for idx in tour:
            cover_indices = self.cover_idx(idx)
            mask[cover_indices]=1
        return torch.where(mask==0)[0]

    def del_redundant(self, tour):
        "遍历所有节点，删除所有冗余的节点：删除后还能覆盖所有的nodes"
        tour_ori = tour
        for i in range(1, tour.shape[0]):       # depot不能删除
            del_tour = torch.concat((tour_ori[:i], tour_ori[i+1:]), dim=0)
            if self.check_cover(del_tour):
                tour_ori = del_tour
        # for node in tour_ori:
        #     del_tour = torch.delete(tour_ori, torch.where(tour_ori==node))
        #     if self.check_cover(del_tour):
        #         tour_ori = del_tour
        return tour_ori


    def mutate(self, tour, best_tour, best_cost):
        r = torch.randint(1, self.num_loc, (1,))        # depot一定在route中，而且不能被删除
        if r not in tour:
            # insert this node to its best place
            best_pos, _ = self.best_add_position(tour, r)
            tour_insert = tour.clone().tolist()
            tour_insert.insert(best_pos, r)
            tour_insert = torch.tensor(tour_insert)
            tour = tour_insert
        else:
            # remove this node, and check feasible
            # del_tour = np.setdiff1d(tour, r)
            # del_tour = torch.delete(tour, torch.where(tour == r))
            idx = torch.where(tour == r)[0]
            del_tour = torch.concat((tour[:idx], tour[idx+1:]), dim=0)
            # nodes_not_select = torch.setdiff1d(torch.range(0, self.num_loc-1), del_tour)
            nodes_not_select = torch.tensor([idx for idx in del_tour if idx not in torch.arange(0, self.num_loc)])
            tour = self.feasible_process(del_tour, nodes_not_select)
        cost_now =  self.dist(tour)
        if cost_now < best_cost:
            best_tour = tour
            best_cost = best_cost
        return tour, best_tour, best_cost

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


    def add_uncovered_nodes(self, end_of_tour, uncovered):
        '''
            先按照最短路径加上uncovered节点
            加一个后更新uncovered，如果提前都覆盖就停止再加节点
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
    
    def forward_single(self, tour=None, stop_cost = -1e6):

        if tour is None:
            tour = self.init_solution()
            # print("Initial Solution!!")

        ids = torch.arange(0, self.num_loc)

        best_cost = self.dist(tour)
        best_tour = tour.to(self.device)
        iter_no_change = 0

        for i in range(self.max_iters):
            nodes_not_select = torch.tensor([idx for idx in ids if idx not in tour])      
            # delete nodes
            del_num = int(tour.shape[0]*self.del_perc)
            del_probs = self.delete_nodes_probs(tour)
            samples = torch.multinomial(del_probs, num_samples=del_num, replacement=False).to("cpu")
            del_nodes = tour[samples]
            # deleted_tour = np.setdiff1d(tour, del_nodes)
            # del_inds = [torch.where(tour == node) for node in del_nodes]       # where返回节点下标， 得到要删除的节点的所有下标
            # deleted_tour = torch.delete(tour, del_inds)
            deleted_tour = torch.tensor([idx for idx in tour if idx not in del_nodes])
            # feasible process
            feasible_tour = self.feasible_process(deleted_tour, nodes_not_select)
            # del redundant nodes
            clean_tour = self.del_redundant(feasible_tour)
            # 2opt to get the shorted path
            clean_tour = self.ls_2opt.LS_2opt(clean_tour, 100)

            cost_now = self.dist(clean_tour)

            # diverse
            if self.if_diverse:
                if cost_now <= best_cost * (1 + self.a):
                    tour = clean_tour
                    # print("replace start-search tour, best tour/cost don't change")
                    if cost_now < best_cost:
                        best_tour = tour
                        best_cost = cost_now
                        iter_no_change = 0
                        # print("replace All")
                else:
                    tour = best_tour
                    iter_no_change = iter_no_change + 1
            else:
                if cost_now < best_cost:
                    # print("replace All")
                    tour = clean_tour
                    best_tour = tour
                    best_cost = cost_now
                else:
                    iter_no_change = iter_no_change + 1

            # mutate
            if self.if_mutate:
                if iter_no_change > self.mutate_iters:
                    tour, best_tour, best_cost = self.mutate(tour, best_tour, best_cost)
            # if iter_no_change > loc.shape[0]:
            if iter_no_change > 50:
                break
            # log
            if self.print_enable:
                if i % 40 == 0:
                    print("Iteration %d, Current distance: %2.3f" % (i, best_cost))
            if best_cost <= stop_cost:
                return -1e6, best_tour
        
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