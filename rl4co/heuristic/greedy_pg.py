import torch
from rl4co.utils.heuristic_utils import convert_to_fit_npz
import time
import random

class Greedy_pg:
    def __init__(self, td) -> None:
        super(Greedy_pg).__init__()
        '''
        td: TensorDict, after call .reset() function
        td["locs"]: [batch, num_customer, 2], customers and depot(0)
        td["tw_low"]: [batch, num_customer]
        td["tw_high"]: [batch, num_customer]
        td["cost"]: [batch, num_customer],  expected cost
        td["stochastic_cost"] : [batch, num_customer],  real cost
        td["attack_prob"]: [batch, num_customer], 
        td["real_prob"]: [batch, num_customer], real prob decide stochastic cost
        td["maxtime"]:  
        td["adj"], 
        td["weather"]: [batch, 3], 

        td["action_mask"]: [batch, num_customer], 0 is not allowed i.e. False; others True
        td["reward"]:  [batch, ], 0
        
        arrive_times: route中每个点的到达时间, [batch, num_customer]; 不在路径中的=0, depot也=0
        wait_times: route中每个点的等待时间, [batch, num_customer]; 不在路径中的=0, depot也=0
        maxshift: init 1e5, 每个点可平移的最大时间； 如果是新加入的
        forward(): call this to get all solutions, a nested list, call convert_to_fit_npz as for savez
        '''
        
        if td is not None:
            self.td = td
            self.num_loc = self.td["locs"].shape[1]
            self.locs = self.td["locs"]
            self.cost = self.td["cost"]
            self.stochastic_cost = self.td["stochastic_cost"]
            self.adj = self.td["adj"]
            self.tw_low = self.td["tw_low"]
            self.tw_high = self.td["tw_high"]
            self.maxtime = self.td["maxtime"]
            self.real_prob = self.td["real_prob"]
            self.attack_prob = self.td["attack_prob"]
            
            self.batch_size = self.td["locs"].shape[0]
        else:
            print("get no data!")

        self.instance_idx = 0
        self.device = "cpu"
        #torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # arrive wait maxshift 只有加入route后才更新，且一定要更新

        self.beta = 1.
        self.alpha = 1.

    def forward(self):
        
        rewards = []
        routes = []

        st = time.time()
        for i in range(self.batch_size):
            print(f"search for data {i}")
            cost, solution, all_info = self.forward_single()
            self.instance_idx += 1
            rewards.append(cost)
            routes.append(solution)
            # print(all_info)
        
        est = time.time()
        print(f"total time: {est - st}, mean time: {(est - st) / self.batch_size}")

        routes = convert_to_fit_npz(routes)

        mean_ = sum(rewards) / len(rewards)
        squared_deviations = [(value - mean_) ** 2 for value in rewards]
        var_ = sum(squared_deviations) / len(rewards)

        return {
            "routes": routes,
            "rewards": rewards,
            "mean reward": mean_,
            "var reward": var_,
            "time": est - st,
        }
    
    def forward_single(self):
        '''
        i: i th data, start from 0
        '''
        self.this_maxtime = int(self.maxtime[self.instance_idx][0])
        self.remaining_time = self.this_maxtime
        self.rem_pos_idx = list(range(self.num_loc))
        self.rem_pos  = self.locs[self.instance_idx, self.rem_pos_idx, :]
        self.curr_node = 0
        curr_route = [0]
        all_route = []
        self.curr_time = 0
        rem_tw_low = self.tw_low[self.instance_idx, self.rem_pos_idx]
        rem_tw_high = self.tw_high[self.instance_idx, self.rem_pos_idx]
        
        
        while self.remaining_time > 0 and len(self.rem_pos) > 0:
            rem_dist = self.adj[self.instance_idx, self.curr_node, self.rem_pos_idx]    # [剩余点数长]
            
            # 可行点的严格条件： 总时间必须在限制内
            arrive_time = torch.ones_like(rem_tw_low) * self.curr_time + rem_dist
            feasible = (arrive_time < rem_tw_high) * (arrive_time > rem_tw_low)
            if not feasible.any():
                # 放款限制
                future = arrive_time < rem_tw_high
                future_dist = rem_dist[future]
                future_tw_low = rem_tw_low[future]
                if not future.any():
                    # 没有下一个点
                    break
                # 选择tw_low最小的
                min_id = future_tw_low.argmin()
                time_to_min = future_dist[min_id]
                wait_time = future_tw_low[min_id] - time_to_min - self.curr_time
                self.remaining_time -= wait_time
                self.curr_time += wait_time
            
            h = self.get_heuristics(rem_tw_high, rem_tw_low)
            a_star = (torch.exp(-self.beta*rem_dist) + self.alpha*h) * feasible

            best = a_star.argmax()
            elapsed = rem_dist[best]
            self.remaining_time -= elapsed
            self.curr_time += elapsed
            if self.curr_time >= self.this_maxtime:
                break
            all_route.append((self.rem_pos[best].cpu().tolist(), elapsed, self.curr_time.cpu(), self.rem_pos_idx[best]))
            curr_route.append(self.rem_pos_idx[best])
            self.curr_node = self.rem_pos_idx[best]
            self.rem_pos_idx.remove(self.curr_node)
            self.rem_pos = self.locs[self.instance_idx, self.rem_pos_idx, :]
            rem_tw_low = self.tw_low[self.instance_idx, self.rem_pos_idx]
            rem_tw_high = self.tw_high[self.instance_idx, self.rem_pos_idx]

        reward = self.get_real_reward(curr_route).cpu()
        return reward, curr_route, all_route
    
    def get_heuristics(self, rem_tw_high, rem_tw_low):
        # higher h get more reward
        h = []
        for i in range(len(self.rem_pos_idx)):

            duration = rem_tw_high - rem_tw_low
            curr_tw_low = rem_tw_low[i]
            curr_tw_high = rem_tw_high[i]
            min_tw_high = torch.where(curr_tw_high < rem_tw_high, curr_tw_high, rem_tw_high)
            max_tw_low = torch.where(curr_tw_low > rem_tw_low, curr_tw_low, rem_tw_low)
            window_overlap = 1 -  abs(min_tw_high - max_tw_low)/ duration

            curr_dist = self.adj[self.instance_idx, self.rem_pos_idx[i], self.rem_pos_idx]  ## i node到别的点的距离, [rem点个数]
            tmp = curr_dist * window_overlap
            # 均值，exp是当前节点的h值
            h_i = torch.exp(-self.beta*tmp.mean())
            # 考虑prize: attack_prob*cost
            expect_prize = self.attack_prob[self.instance_idx, self.rem_pos_idx[i]] * self.cost[self.instance_idx, self.rem_pos_idx[i]]
            h_i += expect_prize
            h.append(h_i)
        
        return torch.tensor(h, device=self.device)
            
    def get_real_reward(self, route):
        '''
        collect :defended reward - undefended attack's costs
        '''
        # 所有路径上点的costs
        # print(f"cost", self.cost[self.instance_idx])

        reward = (self.stochastic_cost[self.instance_idx, torch.tensor(route).to(self.td.device)] * \
            self.real_prob[self.instance_idx, torch.tensor(route).to(self.td.device)]).sum()
        # print(f"reward:", reward)
        # 有attack但是未拜访的
        # undefend = torch.ones(self.num_loc, device=self.td.device, dtype=torch.bool)
        # undefend[torch.tensor(route).to(self.td.device)] = False
        # costs = (self.stochastic_cost[self.instance_idx] * undefend).sum(-1)
        # reward -= costs
        return reward
        






# calculate_best_insert_position(1, [3, 5, 7, 6, 9])