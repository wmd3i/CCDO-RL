import torch
from rl4co.utils.heuristic_utils import convert_to_fit_npz
import time
import random

class LocalSearch_pg:
    def __init__(self, td) -> None:
        super(LocalSearch_pg).__init__()
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
        # arrive wait maxshift 只有加入route后才更新，且一定要更新
        self.arrive_times = torch.zeros(self.batch_size, self.num_loc).to(td.device)
        self.wait_times = torch.zeros(self.batch_size, self.num_loc).to(td.device)
        self.maxshift = torch.ones(self.batch_size, self.num_loc).to(td.device) * 1e5
        self.remove_numbers_base = int(self.num_loc / 3)   

        # shake
        self.remove_placeS = 1
        self.remove_secutiveR = 1 
    def forward(self):
        
        rewards = []
        routes = []

        st = time.time()
        for i in range(self.batch_size):
            print(f"search for data {i}")
            cost, solution = self.forward_single()
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
            "mean reward": mean_,
            "var reward": var_,
            "time": est - st,
        }
    
    def forward_single(self):
        '''
        i: i th data, start from 0
        '''
        no_improve_times = 0
        curr_reward = 0
        curr_route = [0]
        best_reward = -1e5
        best_route = None

        while no_improve_times < 150:
            # print(f"iter:-----------{no_improve_times}-------------")
            curr_route = self.insert(curr_route)       # 到达local optim
            curr_reward = self.get_reward(curr_route)
            # print("before: insert", curr_route)
            if curr_reward > best_reward:
                best_reward = curr_reward
                best_route = curr_route.copy()

                self.remove_secutiveR = 1
                no_improve_times = 0
                # print("new best reward:", best_reward)
                # print("new best route:", best_route)
            else:
                no_improve_times += 1

            curr_route = self.shake(curr_route)     # 如果S+R超出范围，把所有的都切掉，不从头开始
            # print("after: shake" ,curr_route)

            # show route
            # self.show_route(curr_route)
            
            # update S R
            self.remove_placeS += self.remove_secutiveR
            if self.remove_placeS > len(curr_route) - 1:
                self.remove_placeS = 1
            
            self.remove_secutiveR += 1
            if self.remove_secutiveR > self.remove_numbers_base:
                self.remove_secutiveR = 1

        final_reward = self.get_real_reward(best_route)

        return final_reward, best_route
    
    def get_reward(self, route):
        '''
        collect :defended reward - undefended attack's costs
        '''
        # 所有路径上点的costs
        # print(f"cost", self.cost[self.instance_idx])

        reward = (self.cost[self.instance_idx, torch.tensor(route).to(self.td.device)] * \
            self.attack_prob[self.instance_idx, torch.tensor(route).to(self.td.device)]).sum()
        # print(f"reward:", reward)
        # 有attack但是未拜访的
        # undefend = torch.ones(self.num_loc, device=self.td.device, dtype=torch.bool)
        # undefend[torch.tensor(route).to(self.td.device)] = False
        # costs = (self.stochastic_cost[self.instance_idx] * undefend).sum(-1)
        # reward -= costs
        return reward

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
    def show_route(self, route):
        fore = 0
        for n in route[1:]:
            print(f"from {fore} to {n}, distance {self.adj[self.instance_idx, fore, n]:.2f} in {self.arrive_times[self.instance_idx, n]:.2f}, in low tw {self.tw_low[self.instance_idx, n]:.2f}, high tw: {self.tw_high[self.instance_idx, n]:.2f}")
            fore = n

    def insert(self, route):
        done = False
        
        while not done:
            route, n = self.insert_once(route)
            # print(route)
            # print(f"new added node: {n},  in {self.arrive_times[self.instance_idx, n]:.2f}, in high tw: {self.tw_high[self.instance_idx, n]:.2f}")
            # 没有可行的node， 或者没有足够的时间回到depot， 结束
            done = (n == -1) | (self.is_done(route))

        # show 
        # self.show_route(route)
        # print("maxtime", self.maxtime[self.instance_idx][0])

        return route
    

    def is_done(self, route):
        end_node = route[-1]
        tour_time = self.arrive_times[self.instance_idx, end_node]
        curr_maxtime = self.maxtime[self.instance_idx][0]
        return self.check_if_back(end_node, tour_time, curr_maxtime)
    
    def check_if_back(self, curr_node, tour_time, curr_maxtime):
        ## 不能回到depot的都结束done
        distance_back_to_depot = self.adj[self.instance_idx, curr_node, 0]
        back_to_depot = (tour_time + distance_back_to_depot) > curr_maxtime
        return back_to_depot
    
    def insert_once(self, route):
        '''
        route: list, 当前的路径
            每次插入一个最佳节点，先计算每个novisit节点的最佳位置，再跟据ratio得到最佳插入的节点 
        '''
        no_visited = [node for node in range(self.num_loc) if node not in route]
        if not no_visited:
            # print("no nodes can be inserted")
            return route, -1
        
        nodes_insert_info = []
        shifts = []
        no_visited_feasible = []
        for nv_node in no_visited:
            inserted_after_node, arrive_j, wait_j, shift_j = self.calculate_best_insert_position(nv_node, route.copy())

            if inserted_after_node == -1:
                # print(f"\t\t{nv_node} can not be inserted for no feasible position")
                continue
            else:
                nodes_insert_info.append((inserted_after_node, arrive_j, wait_j, shift_j))
                shifts.append(shift_j)
                no_visited_feasible.append(nv_node)

        if not no_visited_feasible:
            # print("\tno feasible nodes now")
            return route, -1

        # ratio最大的插入
        shifts = torch.tensor(shifts).to(self.td.device)
        ratio = self.cost[self.instance_idx, torch.tensor(no_visited_feasible).to(self.td.device)] **2 / shifts        # self.cost
        best_insert_node_idx = torch.argmax(ratio)       # 返回idx不是element
        best_insert_node = no_visited_feasible[int(best_insert_node_idx)]
        
        inserted_after_node = nodes_insert_info[best_insert_node_idx][0]
        inserted_arrive_j = nodes_insert_info[best_insert_node_idx][1]
        inserted_wait_j = nodes_insert_info[best_insert_node_idx][2]
        inserted_shift_j = nodes_insert_info[best_insert_node_idx][3]
        idx = route.index(inserted_after_node)
        route.insert(idx+1, best_insert_node)

        # 更新插入的node的arrive, wait
        self.arrive_times[self.instance_idx, best_insert_node] = inserted_arrive_j
        self.wait_times[self.instance_idx, best_insert_node] = inserted_wait_j

        later_nodes = route[idx+2:]
        if len(later_nodes) > 0:
            # if not insert at end, update # 更新插入的node的后面的arrive, wait, maxshift
            tmp_shift = inserted_shift_j
            for n in later_nodes:
                self.arrive_times[self.instance_idx, n] += tmp_shift
                orig_wait = self.wait_times[0, n].clone()
                tmp = torch.clamp(self.wait_times[self.instance_idx, n] - tmp_shift, min=0)
                self.wait_times[self.instance_idx, n] = tmp
                shift_k = torch.clamp(tmp_shift - orig_wait, min=0)
                if abs(shift_k - 0) < 1e-6:     # 时间平移已经消除，后面的不用在平移
                    break
                self.maxshift[self.instance_idx, n] -= shift_k
                tmp_shift = shift_k
                        
        # 更新插入node的maxshift
        if idx + 2 > len(route) -1: # 插入到route尾部
            next_node = -1
        else:
            next_node = route[idx+2]
        self.maxshift = self.update_maxshift(best_insert_node, next_node, inserted_arrive_j)
       
        # 更新插入node前面的节点的maxshift
        pre_node = route[:idx+1]
        for i in range(len(pre_node) - 1):      # depot不用更新
            update_idx = len(pre_node) - i - 1
            update_next_idx = update_idx + 1
            update_node = route[update_idx]
            update_node_next = route[update_next_idx]
            self.maxshift = self.update_fore_maxshift(update_node, update_node_next)

        return route, best_insert_node
    
    def update_maxshift(self, update_node, update_node_next, arrive_j):
        '''
        更新插入节点的maxshift， 论文公式11， 到达时间为arrive_j
        '''
        if update_node_next == -1:  # 如果是插入route尾部，maxshift = 时间窗 - 到达时间
            self.maxshift[self.instance_idx, update_node] = self.tw_high[self.instance_idx, update_node] - arrive_j     
        else:
            next_tmp = self.wait_times[self.instance_idx, update_node_next] + self.maxshift[self.instance_idx, update_node_next]
            self.maxshift[self.instance_idx, update_node] = torch.min(self.tw_high[self.instance_idx, update_node] - arrive_j, next_tmp)
        return self.maxshift
    
    def update_fore_maxshift(self, update_node, update_node_next):
        '''
        更新插入节点的前面节点的maxshift, 论文公式11，使用自己的arrive times
        '''
        if update_node_next == -1:  # 如果是插入route尾部，maxshift = 时间窗 - 到达时间
            curr_arrive = self.arrive_times[self.instance_idx, update_node].clone()
            self.maxshift[self.instance_idx, update_node] = self.tw_high[self.instance_idx, update_node] - curr_arrive 
        # min(下个node的wait+maxshift)
        else:
            next_tmp = self.wait_times[self.instance_idx, update_node_next] + self.maxshift[self.instance_idx, update_node_next]
            curr_arrive = self.arrive_times[self.instance_idx, update_node].clone()
            self.maxshift[self.instance_idx, update_node] = torch.min(self.tw_high[self.instance_idx, update_node] - curr_arrive, next_tmp)
        return self.maxshift


    def calculate_best_insert_position(self, inserted_node, route):
        '''
        arrive_time: route中每个点的到达时间, [batch, num_customer]
        wait_time: route中每个点的等待时间, [batch, num_customer]
        将inserted_node插入到route中,返回插入route的最佳位置: shift最小的位置
        '''
        # 张量计算
        device = "cpu"
        if isinstance(route, list):
            route = torch.tensor(route).to(device)

        next_node = torch.tensor(route).roll(-1)    # 向左移动
        # next_node[-1] = torch.tensor(route)[-1]     # 最后节点的next赋值为自身， 下面计算距离时 = 0

        arrive = self.arrive_times[self.instance_idx, :].gather(0, route) + \
                    self.wait_times[self.instance_idx, :].gather(0, route) + \
                        self.adj[self.instance_idx, route, inserted_node]      # [len(route)], 插入该node后面, j自身的arrive时间
        

        wait = torch.clamp(self.td["tw_low"][self.instance_idx, inserted_node] - arrive, min=0)  # 插入该node后面, j自身的wait时间
        

        shifts = wait + self.adj[self.instance_idx, route, inserted_node] +\
              self.adj[self.instance_idx, inserted_node, next_node] - \
                  self.adj[self.instance_idx, route, next_node]        # [len(route)]， 插入该node后面, 其他点需要shift的时间
        
        # 插入可行性
        under_tw_high = (arrive < self.tw_high[self.instance_idx, inserted_node])  # 必须在时间窗内
        time_limit = self.wait_times[self.instance_idx, next_node] + self.maxshift[self.instance_idx, next_node]  
        feasible = under_tw_high * (shifts <= time_limit)

        if (~feasible).all():
            # print("no feasible nodes")
            return -1, 0, 0, 0
        shifts_tmp = torch.zeros(self.num_loc).to(device)        # 插入到node idx后，后面的平移时间 
        shifts_tmp = shifts_tmp.scatter(0, route, torch.tensor(shifts))     # 加到inserted_node以后的node上，如果是最后一个点，

        #根据shifts得到inserted_node插入到route中的最佳位置： min shifts的pos和shift
        shifts_tmp = shifts_tmp.gather(0, route)
        shifts_tmp = torch.where(feasible, shifts_tmp, torch.tensor(1e10).to(device))
        
        prob = 0.5
        if shifts_tmp.shape[0] > 1:     # 如果有多于两个选择，以一定概率随机选一个
            if random.random() < prob:
                best_insert_idx = random.choice(range(shifts_tmp.shape[0]))
            else:
                best_insert_idx = torch.argmin(shifts_tmp)
        else:
            best_insert_idx = torch.argmin(shifts_tmp)

        best_insert_pos = route[best_insert_idx]


        arrive_this = arrive[best_insert_idx]
        wait_this = wait[best_insert_idx]
        shift_this = shifts_tmp[best_insert_idx]
        return best_insert_pos, arrive_this, wait_this, shift_this
    

    def shake(self, route):
        
        fore_route = route.copy()[:self.remove_placeS]
        later_route =  route.copy()[self.remove_placeS+self.remove_secutiveR:]
        new_route = fore_route + later_route
        
        if not later_route:
            pass
        else:
            # 更新后面的node：arrive, wait, maxshift
            # 先更新arrive， 再更新wait
            curr_node = later_route[0]
            before_node = fore_route[-1]
            before_arrive = self.arrive_times[self.instance_idx, before_node].clone() + self.wait_times[self.instance_idx, before_node].clone()
            arrive = before_arrive + \
                    self.adj[self.instance_idx, before_node, curr_node]
            
            orig_arrive = self.arrive_times[self.instance_idx, curr_node].clone()
            self.arrive_times[self.instance_idx, curr_node] = arrive
            self.wait_times[self.instance_idx, curr_node] = torch.clamp(self.td["tw_low"][self.instance_idx, curr_node] - arrive, min=0.)
            
            shift = torch.max(self.tw_low[self.instance_idx, curr_node], arrive) - orig_arrive
            # 更新自己的maxshift。= 原来的maxshift + 现在往前移动的时间
            orig_shift = self.maxshift[self.instance_idx, curr_node].clone()
            self.maxshift[self.instance_idx, curr_node] = orig_shift + orig_arrive - arrive
            shift = shift * (orig_arrive > self.tw_low[self.instance_idx, curr_node])       # 如果原来就等待，后面无需平移
            


            for n in later_route[1:]:
                if not shift:   # 直到shift = 0
                    break
                before_node = curr_node
                curr_node = n
                orig_arrive = self.arrive_times[self.instance_idx, curr_node].clone()
                arrive = self.arrive_times[self.instance_idx, curr_node] + shift
                wait = torch.clamp(self.td["tw_low"][self.instance_idx, curr_node] - arrive, min=0)
                self.arrive_times[self.instance_idx, curr_node] = arrive
                self.wait_times[self.instance_idx, curr_node] = wait
                shift = torch.max(self.tw_low[self.instance_idx, curr_node], arrive) - orig_arrive
                shift = shift * (orig_arrive > self.tw_low[self.instance_idx, curr_node])
                # 更新maxshift: 直接在原来上面加平移的非零shift
                orig_shift = self.maxshift[self.instance_idx, curr_node].clone()
                self.maxshift[self.instance_idx, curr_node] = orig_shift + orig_arrive - arrive
        
        # 更新前面nodes的maxshift
        
        for i in range(len(fore_route) - 1):      # depot不用更新
            update_idx = len(fore_route) - i - 1
            update_next_idx = update_idx + 1
            update_node = new_route[update_idx]

            if update_next_idx == len(new_route):   # 最后一个点
                update_node_next = -1
            else:
                update_node_next = new_route[update_next_idx]
            self.maxshift = self.update_fore_maxshift(update_node, update_node_next)

        return new_route
        






# calculate_best_insert_position(1, [3, 5, 7, 6, 9])