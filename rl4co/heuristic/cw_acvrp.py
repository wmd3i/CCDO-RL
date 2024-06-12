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
class CW_acvrp:
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
        self.pair_wise = None       # [batch, num_customer, num_customer]
        self.num_customer = 0
        self.data_nodes = None      # [batch, num_customer, 2], distance to depot, demand
        self.savings = dict()
        self.sorted_savings = None
        self.routes = list()
        self.capacity = 1        # define by env
        self.print_enable = False
        
    def __node_data(self):
        # get node data in coordinate (x,y) format from envs 
        # node idx, distance to depot, demand
        depot = self.td["locs"][:, 0:1, :]
        distance_to_depot = get_distance(self.td["locs"][:, 1:, :], depot)     #[batch, num_loc]
        demand = self.td["demand"]      # expected demand, [batch, num_loc, 1]
        self.data_nodes = torch.concatenate((distance_to_depot[..., None], demand[..., None]), -1)        #[batch, num_loc, 2]
        
    def __pairwise(self):
        # read pairwise distance: 
        self.num_customer = self.td["locs"].size(1) - 1     # substract depot
        # self.pair_wise = torch.zeros((num_customers, num_customers))
        customer_locs = self.td["locs"][:, 1:, :]
        loc_left_x = customer_locs[..., 0][..., None].repeat(1, 1, self.num_customer)  # [batch, num_locs, num_locs]
        loc_left_y = customer_locs[..., 1][..., None].repeat(1, 1, self.num_customer)  # [batch, num_locs, num_locs]
        loc_right_x = torch.transpose(loc_left_x, -1, -2)      # [batch, num_locs, num_locs]
        loc_right_y = torch.transpose(loc_left_y, -1, -2)
        # [batch, num_locs, num_locs]
        self.pair_wise = torch.sqrt((loc_left_x - loc_right_x)**2 + (loc_left_y  - loc_right_y)**2)
            
    
    def __get_savings(self):
        # calculate savings for each link
        for r in range(1, self.num_customer+1, 1):      # start from 1, to num_cust
            for c in range(r, self.num_customer+1, 1):
                if int(c) != int(r):            
                    a = max(int(r), int(c))
                    b = min(int(r), int(c))
                    key = '(' + str(a) + ',' + str(b) + ')'
                    self.savings[key] = self.data_nodes[:, r-1, 0] + self.data_nodes[:, c-1, 0] - self.pair_wise[:, r-1, c-1]   #[batch]


    @staticmethod
    def get_node(link):
    # convert link string to link list to handle saving's key, i.e. str(10, 6) to (10, 6)
        link = link[1:]
        link = link[:-1]
        nodes = link.split(',')
        return [int(nodes[0]), int(nodes[1])]

    @staticmethod
    def which_route(link, routes_single):
    # determine 4 things:
    # 1. if the link in any route in routes -> determined by if count_in > 0
    # 2. if yes, which node is in the route -> returned to node_sel
    # 3. if yes, which route is the node belongs to -> returned to route id: i_route
    # 4. are both of the nodes in the same route? -> overlap = 1, yes; otherwise, no
    
        # assume nodes are not in any route
        node_sel = list()
        i_route = [-1, -1]
        count_in = 0
        
        for route in routes_single:
            for node in link:
                try:
                    route.index(node)
                    i_route[count_in] = routes_single.index(route)
                    node_sel.append(node)
                    count_in += 1
                except:
                    pass
                    
        if i_route[0] == i_route[1]:
            overlap = 1
        else:
            overlap = 0
            
        return node_sel, count_in, i_route, overlap
    
        
    def _sum_cap(self, i, route):
        # sum up to obtain the total used capacity belonging to a route
        sum_cap = 0
        for node in route:
            sum_cap += self.data_nodes[i, node-1, 1]      # get demand of node in data i
        return sum_cap

    @staticmethod
    def interior(node, route):
    # determine if a node is interior to a route
        try:
            i = route.index(node)
            # adjacent to depot, not interior
            if i == 0 or i == (len(route) - 1):
                label = False
            else:
                label = True
        except:
            label = False
        
        return label



    @staticmethod
    def merge(route0, route1, link):
    # merge two routes with a connection link
        if route0.index(link[0]) != (len(route0) - 1):
            route0.reverse()
        
        if route1.index(link[1]) != 0:
            route1.reverse()
            
        return route0 + route1

    def _get_reward(self, td: TensorDict, batch_actions):
        '''
        batch_actions: from 1 to num_customers
        '''
        batch_actions = pad_sequence([torch.tensor(actions, device=td["demand"].device, 
                                                   dtype=torch.int64) 
                                      for actions in batch_actions], 
                                     batch_first=True, padding_value=0)
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
        
        st = time.time()
        self.__node_data()
        self.__pairwise()
        self.__get_savings()
        
        for i in range(self.data_nodes.size(0)):
            if self.print_enable:
                print(f"instance {i}")
            single_routes = self.forward_single(i)
            # concat all routes to a route of an instance
            single_routes = [action for routes in single_routes for action in routes]       # actions with multiply depots
            self.routes.append(single_routes)
        
        et = time.time() - st
        rewards = self._get_reward(self.td, self.routes)     #[batch]
        mean_reward = rewards.mean().cpu().item()
        var_ = rewards.var().cpu().item()
        print('------CW-----')
        # print(f'Routes found are:{self.routes}, rewards are {rewards}, mean reward is {mean_reward} ')
        
        routes = convert_to_fit_npz(self.routes)

        return {"routes":routes,
                "rewards": rewards,
                "mean reward": mean_reward,
                "var reward": var_,
                "time": et,
            }

    def forward_single(self, i):
        '''
        i: i th data, start from 0
        '''
        savings_single = {link: value[i] for link, value in self.savings.items()}
        
        # self.sort_savings(savings_single)
        saving_single_sorted = sorted(savings_single.items(), key=lambda s: s[1], reverse=True)
        
        # if there is any remaining customer to be served
        remaining = True
        # record steps
        step = 0
        # get a list of nodes, excluding the depot, start from 1
        node_list = list(range(1, self.num_customer+1, 1))

        self.routes_single = list()
        # run through each link in the saving list
        for link, value in saving_single_sorted:
            step += 1
            if remaining:

                # print('step ', step, ':')

                link = self.get_node(link)
                node_sel, num_in, i_route, overlap = self.which_route(link, self.routes_single)

                # condition a. Either, neither i nor j have already been assigned to a route, 
                # ...in which case a new route is initiated including both i and j.
                if num_in == 0:
                    if self._sum_cap(i, link) <= self.capacity:
                        self.routes_single.append(link)
                        node_list.remove(link[0])
                        node_list.remove(link[1])
                        # print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
                    else:
                        # print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load, so skip this link.')
                        pass

                # condition b. Or, exactly one of the two nodes (i or j) has already been included 
                # ...in an existing route and that point is not interior to that route 
                # ...(a point is interior to a route if it is not adjacent to the depot D in the order of traversal of nodes), 
                # ...in which case the link (i, j) is added to that same route.    
                elif num_in == 1:
                    n_sel = node_sel[0]
                    i_rt = i_route[0]
                    position = self.routes_single[i_rt].index(n_sel)
                    link_temp = link.copy()
                    link_temp.remove(n_sel)
                    node = link_temp[0]

                    cond1 = (not self.interior(n_sel, self.routes_single[i_rt]))
                    cond2 = (self._sum_cap(i, self.routes_single[i_rt] + [node]) <= self.capacity)

                    if cond1:
                        if cond2:
                            # print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', self.routes_single[i_rt], '.')
                            if position == 0:
                                self.routes_single[i_rt].insert(0, node)
                            else:
                                self.routes_single[i_rt].append(node)
                            node_list.remove(node)
                        else:
                            # print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum load, so skip this link.')
                            continue
                    else:
                        # print('\t','For Link ', link, ', node ', n_sel, ' is interior to route ', self.routes_single[i_rt], ', so skip this link')
                        continue

                # condition c. Or, both i and j have already been included in two different existing routes 
                # ...and neither point is interior to its route, in which case the two routes are merged.        
                else:
                    if overlap == 0:
                        cond1 = (not self.interior(node_sel[0], self.routes_single[i_route[0]]))
                        cond2 = (not self.interior(node_sel[1], self.routes_single[i_route[1]]))
                        cond3 = (self._sum_cap(i, self.routes_single[i_route[0]] + self.routes_single[i_route[1]]) <= self.capacity)

                        if cond1 and cond2:
                            if cond3:
                                route_temp = self.merge(self.routes_single[i_route[0]], self.routes_single[i_route[1]], node_sel)
                                temp1 = self.routes_single[i_route[0]]
                                temp2 = self.routes_single[i_route[1]]
                                self.routes_single.remove(temp1)
                                self.routes_single.remove(temp2)
                                self.routes_single.append(route_temp)
                                try:
                                    node_list.remove(link[0])
                                    node_list.remove(link[1])
                                except:
                                    #print('\t', f"Node {link[0]} or {link[1]} has been removed in a previous step.")
                                    pass
                                # print('\t','Link ', link, ' fulfills criteria c), so route ', temp1, ' and route ', temp2, ' are merged')
                            else:
                                # print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load, so skip this link.')
                                continue
                        else:
                            # print('\t','For link ', link, ', Two nodes are found in two different routes, but not all the nodes fulfill interior requirement, so skip this link')
                            continue
                    else:
                        # print('\t','Link ', link, ' is already included in the routes')
                        continue

                for route in self.routes_single: 
                    # print('\t','route: ', route, ' with load ', self.sum_cap(i, route))
                    pass
            else:
                # print('-------')
                # print('All nodes are included in the routes, algorithm closed')
                break

            remaining = bool(len(node_list) > 0)

        # check if any node is left, assign to a unique route
        for node_o in node_list:
            self.routes_single.append([node_o])

        # add depot to the routes
        for route in self.routes_single:
            route.insert(0,0)
            route.append(0)
            
        return self.routes_single
 