from tensordict.tensordict import TensorDict
import torch
from torch.nn.utils.rnn import pad_sequence
from rl4co.utils.heuristic_utils import convert_to_fit_npz
import time
import random

class TabuSearch_acvrp:
    
    tabu_len_min = 30
    tabu_len_max = 50
    def __init__(self, td):
        self.td = td
        self.batch_size = self.td["demand"].size(0)
        self.num_customers = self.td["demand"].size(1)
        
        self.TABU = {}
        # for single instance
        self.single_instance = None
        self.distance_matrix = None
        self.tabu_len = 0
        self.best_sol = None
        self.best_cost = None
        self.print_enable = True
        
    def forward(self):
        '''get solutions of all batch instances:
        search according expected demand, but evaluate it with real demand
        '''
        if self.print_enable:
            print('------Tabu search-----')
        time_start = time.time()
        batch_solutions = list()
        batch_costs = list()
        batch_costs_real = list()       
        for i in range(self.batch_size):
            if self.print_enable:
                print(f"search for data {i}")
            instance_i = self.td[i]
            best_solution, best_cost = self.forward_single(instance_i)
            batch_solutions.append(best_solution)
            batch_costs.append(best_cost)
            
            real_cost = self.get_real_penalty(best_solution, i) + best_cost
            batch_costs_real.append(real_cost)        
        # print(f"best cost of all data are {batch_costs}")
        # print(f"real cost of all data are {batch_costs_real}")
        time_end = time.time()
        print(f"time is {time_end - time_start}")
        batch_solutions = convert_to_fit_npz(batch_solutions)
        
        mean_ = sum(batch_costs_real) / len(batch_costs_real)
        squared_deviations = [(value - mean_) ** 2 for value in batch_costs_real]
        var_ = sum(squared_deviations) / len(batch_costs_real)
        return {
            "routes": batch_solutions,
            "rewards": batch_costs_real,
            "mean reward": mean_,
            "var reward": var_,
            "time": time_end - time_start,
        }
    
    
    def get_real_penalty(self, sol, batch_i):
        
        assert self.is_valid(sol), "this solution is invalid"
        
        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        real_demand_with_depot = torch.cat((-self.td[batch_i]["vehicle_capacity"], self.td[batch_i]["real_demand"]))
        d = []
        for route in sol:
            route_d = []
            for custom in route:
                route_d.append(real_demand_with_depot[custom])
            d.append(route_d)

        used_cap = 0.
        penalty_loc_idx = []
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                used_cap += d[i][j]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
                if used_cap < 0:
                    used_cap = 0.
                     
                if used_cap > self.td[batch_i]["vehicle_capacity"] + 1e-5:
                    # print("Used more than capacity")
                    used_cap = d[i][j]
                    penalty_loc_idx.append(sol[i][j])
                    
        penaltied_cost = self.distance_matrix[0, penalty_loc_idx]
        
        return penaltied_cost.sum() * 2
    
    def _get_distance_matrix(self, instance_td):
        # read pairwise distance: 
        self.num_point = instance_td["locs"].size(0)   
        # self.pair_wise = torch.zeros((num_customers, num_customers))
        customer_locs = instance_td["locs"]     # [num_point]
        loc_left_x = customer_locs[:, 0][..., None].repeat(1, self.num_point)  # [num_locs, num_locs]
        loc_left_y = customer_locs[:, 1][..., None].repeat(1, self.num_point)  # [num_locs, num_locs]
        loc_right_x = torch.transpose(loc_left_x, -1, -2)      # [num_locs, num_locs]
        loc_right_y = torch.transpose(loc_left_y, -1, -2)
        # [num_locs, num_locs]
        self.distance_matrix = torch.sqrt((loc_left_x - loc_right_x)**2 + (loc_left_y  - loc_right_y)**2)
        return self.distance_matrix
  
    def forward_single(self, instance_td: TensorDict, n_iters=2000):
        "get solution of an instance: a non-nested list"
        self.single_instance = instance_td
        self.distance_matrix = self._get_distance_matrix(instance_td)        # [num_p, num_p] ,symatric 
        self.best_sol = self.generate_greedy_solution(instance_td)
        self.best_cost = self.get_cost(self.best_sol)
        ##
        while n_iters > 0:
            self.tabu_len = random.randint(TabuSearch_acvrp.tabu_len_min, TabuSearch_acvrp.tabu_len_max)
            searched_nb_num = 0
            
        
            nb_structure_idx = random.randint(0, 3)
            while searched_nb_num < 4:
                # search in a nbhood
                # self.candidate = self.find_neighborhood(nb_structure_idx)
                candidate, candidate_reward, candi_move = self.find_neighborhood(nb_structure_idx)
                
                
                if candidate_reward < self.best_cost:
                    self.best_sol = candidate
                    self.best_cost = candidate_reward
                    # increase tabu_length and continue search in this neighbor structure
                    self.tabu_len += 1
                    self.tabu_len = min(self.tabu_len, TabuSearch_acvrp.tabu_len_max)
                    
                    # Add candidate to tabu
                    self.TABU[candi_move] = self.tabu_len
                else:
                    # else decrease tabu_length and 
                    self.tabu_len -= 1
                    self.tabu_len = max(self.tabu_len, TabuSearch_acvrp.tabu_len_min)
                    # search next neighbot hood
                    nb_structure_idx += 1
                    nb_structure_idx %= 4
                    searched_nb_num += 1
                    self.TABU.clear()
                    
                
                
                 # Remove moves from tabu list： 每个禁忌值-1， 去掉记录值减为0的禁忌值, 
                moves_to_delete = []
                for move, i in self.TABU.items():
                    if i == 0:
                        moves_to_delete.append(move)
                    else:
                        self.TABU[move] -= 1
                        
                for s in moves_to_delete:
                    del self.TABU[s]

            n_iters -= 1
        
       
        return self.best_sol, self.best_cost
    
    def get_cost(self, solution):
        "positive, sum of travel costs, smaller then better"
        cost = 0.
        for route in solution:
            for p in range(len(route) - 1):
                cost += self.distance_matrix[route[p]][route[p+1]]
        return cost
    
    def find_neighborhood(self, nb_idx):
        
        assert nb_idx < 4, "wrong neighborhood structure index"
        if nb_idx == 0:
            nbs, moves = self.__search_specific()
        elif nb_idx == 1:
            nbs, moves = self.__two_opt()
        elif nb_idx == 2:
            nbs, moves = self.__swap_oper()
        else:
            nbs, moves = self.__reallocate()

        # find best solution in nbs
        temp_move = None
        best_in_nbs = None
        best_nbs_cost = 1e3     # if none nbs, thus not better cost
        for nb, move in zip(nbs, moves):
            if move not in self.TABU:
                best_in_nbs = nb
                best_nbs_cost = self.get_cost(best_in_nbs)
                temp_move = move
                break
            
        for nb, move  in zip(nbs[1:], moves[1:]):        # 和领域内其他解比较，找到更好的候选解
                nb_cost = self.get_cost(nb)
                if nb_cost < best_nbs_cost:
                    if move not in self.TABU:       # 不仅要更有，还得是不在禁忌表里面的
                        best_in_nbs = nb
                        best_nbs_cost = nb_cost
                        temp_move = move
                    else:
                        # Aspiration criteria
                        if nb_cost < best_nbs_cost:
                            best_in_nbs = nb
                            best_nbs_cost = nb_cost
                            temp_move = move
        return best_in_nbs, best_nbs_cost, temp_move
    
    
    def is_valid(self, sol: list):
        '''
            check if visits every customer once
        '''
        lst_sol = [node for route in sol for node in route]
        lst_sol = sorted(lst_sol)
        rg = list(range(1, self.num_customers + 1))
        valid = rg == lst_sol[-self.num_customers:]
        return valid
    
    def satisfy_demand(self, route: list):
        '''
            check if route satisfy demand: not exceed capacity
        '''
        used_capacity = 0.
        for customer in route[1:-1]:
            used_capacity += self.single_instance["demand"][customer - 1]
        
        return used_capacity < 1.
        
    def __search_specific(self):
        '''
        random select a customer in route l, insert to another route that can satisfy constraint.
        loop for every route l
        return:
            moves: nested list, (idx of seleted customer route, selected_customer, idx of insert customer route)
        '''
        neighborhood = []
        moves = []
        move = None 
        # iterate every route to selet and insert
        sol = [route[:] for route in self.best_sol]
        for i in range(len(self.best_sol)):
            if len(self.best_sol[i]) <=2 :
                continue
            else:
                selected_customer = random.choice(self.best_sol[i][1:-1])
                
                # iterate other all routes until choose a valid insert
                sol = [route[:] for route in self.best_sol]
                sol[i].remove(selected_customer)
                j = 0
                while j < len(sol):
                    if j != i:      # can not be selected route
                        curr_route = sol[j][:]
                        # insert successfully then break
                        
                        curr_route.insert(-1, selected_customer)
                        if self.satisfy_demand(curr_route):
                            # return new solution
                            
                            sol[j] = curr_route
                            move = (i, selected_customer, j)
                            break
                        # else continue
                        j += 1
                        if j < len(sol):
                            # print("none of a route can add this customer")
                            pass
                    else:
                        j += 1
                
                if j < len(sol):        
                    # check sol valid
                    assert self.is_valid(sol), "this solution is not valid, visited some customer twice"
                    # record move and nb
                    neighborhood.append(sol)
                    moves.append(move)
                
        
        return neighborhood, moves

    def __two_opt(self):
        """random select 2 customers in a route and flips them
        loop every route
        move: (route idx, min custom idx, max custom idx)
        """
        neighborhood = []
        moves = []
        move = None 
        for i in range(len(self.best_sol)):
            sol = [route[:] for route in self.best_sol]
            flipped = sol[i][:]
            idxs = list(range(1, len(flipped)-1))   # remove two depots
            if len(idxs) <= 1:
                continue
            else:
                a = random.choice(idxs)
                idxs.remove(a)
                b = random.choice(idxs)
                min_idx = min(a, b)
                max_idx = max(a, b)
                # flipped[min_idx:max_idx+1] = reversed(flipped[min_idx:max_idx+1])
                flipped[min_idx:max_idx+1] = flipped[min_idx:max_idx+1][::-1]
                
                sol[i] = flipped
                move = (i, min_idx, max_idx)       # (route idx, min custom idx, max custom idx)
                neighborhood.append(sol)
                moves.append(move)
            
        return neighborhood, moves
            
    def __swap_oper(self):
        """random select 2 customers in a route and change them
        loop every route
        move: (route idx, min custom idx, max custom idx)
        """
        neighborhood = []
        moves = []
        move = None 
        for i in range(len(self.best_sol)):
            sol = [route[:] for route in self.best_sol]
            changed = sol[i][:]
            idxs = list(range(1, len(changed)-1))   # remove two depots
            if len(idxs) <= 1:      # if just one customer, not change it
                continue
            else:
                a = random.choice(idxs)
                idxs.remove(a)
                b = random.choice(idxs)
                min_idx = min(a, b)
                max_idx = max(a, b)
                changed[min_idx], changed[max_idx] = changed[max_idx], changed[min_idx]
                
                sol[i] = changed
                move = (i, min_idx, max_idx)       # (route idx, min custom idx, max custom idx)
                neighborhood.append(sol)
                moves.append(move)
            
        return neighborhood, moves
       
    def __reallocate(self):
        '''
            random select a cut from a route and add it to another route
            loop over every route
            move :(cutted route idx, min cus idx, max cus_idx, inserted route, insert loc)
        ''' 
        neighborhood = []
        moves = []
        move = None 
        for i in range(len(self.best_sol)):
            sol = [route[:] for route in self.best_sol]
            selected_route = sol[i][:]
            idxs = list(range(1, len(selected_route)-1))    # remove two depots
            if len(idxs) <= 1:      # if just one customer, not change it
                continue
            else:
                # get cut from this route
                a = random.choice(idxs)
                idxs.remove(a)
                b = random.choice(idxs)
                min_idx = min(a, b)
                max_idx = max(a, b)
                if min_idx == 1 and max_idx == len(selected_route) - 1:
                    max_idx -= 1        # if select whole cut, substract max 1
                selected_cut = selected_route[min_idx:max_idx+1]    # [min_idx, max_idx+1)
                
                # sol[i] = sol[i][:min_idx] + sol[i][max_idx+1:] 
                sol[i] = sol[i][:min_idx] + sol[i][max_idx+1:]
                # choose another route to insert
                j = 0
                while j < len(sol):
                    if j != i:      # can not be selected route
                        inserted_route = self.best_sol[j][:]
                        # insert location: from 1 to -1
                        insert_loc = random.randint(1, len(inserted_route) - 1) #[a, b]
                        inserted_route = inserted_route[:insert_loc]+ selected_cut+ inserted_route[insert_loc:]
                        # return new solution
                        new_sol = [route[:] for route in sol]       # cutted solution
                        new_sol[j] = inserted_route
                        
                        move = (i, min_idx, max_idx, j, insert_loc)     # (cutted route idx, min cus idx, max cus_idx, inserted route, insert loc)
                        assert self.is_valid(new_sol)
                        neighborhood.append(new_sol)
                        moves.append(move)
                        j += 1
                    else:
                        j += 1

        return neighborhood, moves
                    
            
    def generate_greedy_solution(self, instance_td):
        """greedy initial solution for an instance """
        customers = list(range(1, self.num_customers+1))
        solution = list()

        while len(customers) > 0:
            route = list()
            curr_point = 0
            used_capacity = torch.zeros(self.num_customers, device=instance_td.device)
            next_capacity = torch.zeros(self.num_customers, device=instance_td.device)
            # get satisfied demands customers from rest customers            
            
            # must in rest customer 
            mask = torch.ones((self.num_customers + 1), dtype=torch.bool, device=instance_td.device)
            mask[customers] = False
            negative = 1e3 * torch.ones_like(mask)
            while True:
                
                # must not exceed capacity
                next_capacity = used_capacity + instance_td["demand"]
                mask2 = next_capacity > 1.
                mask = mask | torch.cat((torch.ones(1, dtype=torch.bool, device=instance_td.device), mask2))
                distance = self.distance_matrix[curr_point]     # [num_cus]
                # True —— -1
                distance = torch.where(mask, negative, distance)
                
                if (distance > 1e2).all():       # no satisfied customers
                    solution.append(route)
                    break
                # get closet customer from satisfied ones
                closest = distance.argmin()     # from 0 to num_customer
                route.append(closest)
                used_capacity += instance_td["demand"][closest - 1]
                curr_point = closest
                customers.remove(closest)
                # change mask
                mask[closest] = True
            
        # add depot to start and end
        for route in solution:
            route.insert(0,0)
            route.append(0)
        
        return solution
