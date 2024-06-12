from rl4co.utils.pylogger import get_pylogger
import torch
from torch.nn.utils.rnn import pad_sequence
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from tensordict.tensordict import TensorDict
from typing import Optional
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.routing.cvrp import CVRPEnv
import time
log = get_pylogger(__name__)
from rl4co.envs.routing.env_utils import *
# From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
CAPACITIES = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}

class ACVRPEnv(CVRPEnv):
    """Stochastic Vehicle Routing Problem (CVRP) environment.

    Note:
        The only difference with deterministic CVRP is that the demands are stochastic
        (i.e. the demand is not the same as the real prize).
    """

    name = "acvrp"       # class variable
    _stochastic = True
    generate_method = "modelize" 
    
    env_fixed = True
    
    
                    
    def __init__(self, generate_method = "modelize", env_fix=False, **kwargs):
        super().__init__(**kwargs)
        
        self.generate_method = generate_method
        assert self.generate_method in ["uniform", "modelize", "no_stoch"], "way of generate stochastic data is invalid"

        if env_fix:
            ACVRPEnv.env_fixed = True
            ACVRPEnv.name = "svrp_fix"
        ACVRPEnv.stoch_idx = kwargs.get("stoch_idx")
        


    def get_fix_data(self, graph_pool):
        if ACVRPEnv.env_fixed:
            
            ACVRPEnv.a_loc_with_depot = (
                graph_pool.locs_data
                .to(self.device)
            )
            ACVRPEnv.a_demand = (
            graph_pool.demand_data
            .int()
            + 1
            ).float().to(self.device)
        
    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand and stochastic_demand by capacity to be in [0, 1]
        """
        if ACVRPEnv.name == "svrp_fix":
            return self.load_fixed_data(fpath, batch_size)
        td_load = load_npz_to_tensordict(fpath)
        assert td_load["demand"].max() > 1, "this data range is not larger than [0, 1]"
        td_load.set("demand", td_load["demand"] / td_load["capacity"][:, None])
        td_load.set("stochastic_demand", td_load["stochastic_demand"] / td_load["capacity"][:, None])
        return td_load
    
    def load_fixed_data(self, fpath, batch_size):
        td_load = load_npz_to_tensordict(fpath)
        td_load["depot"] =ACVRPEnv.a_loc_with_depot[:, 0, ...].repeat(batch_size, 1, 1).squeeze(1)
        td_load["locs"]= ACVRPEnv.a_loc_with_depot[:, 1:, ...].repeat(batch_size, 1, 1) 
        td_load["demand"]= ACVRPEnv.a_demand.repeat(batch_size, 1) / td_load["capacity"][:, None]
        if ACVRPEnv.generate_method == "modelize":
            stochastic_demand = get_stoch_var(td_load["demand"].to("cpu"),
                                                    td_load["locs"].to("cpu").clone(), 
                                                    td_load["weather"][:, None, :].
                                                    repeat(1, self.num_loc, 1).to("cpu"),
                                                    None, 
                                                    stoch_idx=ACVRPEnv.stoch_idx).squeeze(-1).float().to(self.device)
        elif ACVRPEnv.generate_method == "uniform":
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        td_load["stochastic_demand"]= stochastic_demand / td_load["capacity"][:, None].to(self.device)
        return td_load
    
    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is False:
            log.warning(
                "Deterministic mode should not be used for SVRP. Use CVRP instead."
            )

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # get the solution's penaltied idx
        loc_idx_penaltied = self.get_penalty_loc_idx(td, actions)   #[batch, num-customer]
        # Gather dataset in order of tour
        depot = td["locs"][..., 0:1, :]
        depot_batch = depot.repeat(1, loc_idx_penaltied.size(1), 1) # [batch,  num_customer, 2]
        
        # get penaltied lcoations
        locs_penalty = td["locs"][..., 1:, :]* loc_idx_penaltied[..., None]       #[batch, num_customer, 2]
        # get 0 pad mask
        posit = loc_idx_penaltied > 0  #[batch, num_customer]
        posit = posit[:,:, None]    #[batch, num_customer, 1]
        posit = posit.repeat(1, 1, 2)   #[batch, num_customer, 2]
        locs_penalty = torch.where(posit, locs_penalty, depot_batch)
        
        locs_ordered = torch.cat([depot, gather_by_index(td["locs"], actions)], dim=1)
        cost_orig = -get_tour_length(locs_ordered)
        cost_penalty = -get_distance(depot_batch, locs_penalty).sum(-1) * 2
        # print(f"new version: orig is {cost_orig}, penalty is {cost_penalty}")
        
        return cost_penalty + cost_orig
    
    
            
    
    @staticmethod
    def get_penalty_loc_idx(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot.
        if capacity is not exceeded, record the loc idx
        return penaltied location idx, [batch, penalty_number]
        
        review:
            return penaltied_idx: [batch, num_customer], if node is penaltied, set 1
        """
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td["demand"].size()
        sorted_pi = actions.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        real_demand_with_depot = torch.cat((-td["vehicle_capacity"], td["real_demand"]), 1)
        d = real_demand_with_depot.gather(1, actions)

        
        start_3 = time.time()
        used_cap = torch.zeros((td["demand"].size(0), 1), device=td["demand"].device)
        penaltied_idx = torch.zeros_like(td["demand"], device=td["demand"].device)      # [batch, num_customer]
        for i in range(actions.size(1)):
            used_cap[:, 0] += d[
                    :, i
                ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            exceed_cap_bool = used_cap[:, 0] > td["vehicle_capacity"][:,0] + 1e-5        # 1 dim
            if any(exceed_cap_bool):
                # print("Used more than capacity")
                exceed_idx = torch.nonzero(exceed_cap_bool)     # [exceed_data_idx, 1]
                penaltied_node = actions[exceed_idx, i] - 1     # [exceed_data_idx, 1], in"actions",customer node start from 1, substract 1 when be idx
                penaltied_idx[exceed_idx, penaltied_node] = 1        # set exceed idx to 1
                used_cap[exceed_idx, 0] = d[exceed_idx, i]
        end_3 = time.time()
        # print(f"time of one loop: {end_3 - start_3}")
        return penaltied_idx
    
    # @profile(stream=open('log_mem_svrp_generate_data.log', 'w+'))  
    def generate_data(self, batch_size, ) -> TensorDict:
        if self.name == "svrp_fix":
            return self.generate_fixed_data(batch_size)
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the demand for nodes except the depot
        # Demand sampling Following Kool et al. (2019)
        # Generates a slightly different distribution than using torch.randint
        demand = (
            torch.FloatTensor(*batch_size, self.num_loc)
            .uniform_(self.min_demand - 1, self.max_demand - 1)
            .int()
            + 1
        ).float().to(self.device)

        # Initialize the weather
        weather = (
            torch.FloatTensor(*batch_size, 3)
            .uniform_(-1, 1)
        ).to(self.device)
        
        #  E(stochastic demand) = E(demand)
        if self.generate_method == "uniform":
            # print(f"generate data by uniform")
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        elif self.generate_method == "no_stoch":
            stochastic_demand = demand.clone()
        elif self.generate_method == "modelize":
            # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

            stochastic_demand = get_stoch_var(demand.to("cpu"),
                                                   locs_with_depot[..., 1:, :].to("cpu").clone(), 
                                                   weather[:, None, :].
                                                   repeat(1, self.num_loc, 1).to("cpu"),
                                                   None,
                                                   stoch_idx=ACVRPEnv.stoch_idx).squeeze(-1).float().to(self.device)

        # Support for heterogeneous capacity if provided
        if not isinstance(self.capacity, torch.Tensor):
            capacity = torch.full((*batch_size,), self.capacity, device=self.device)
        else:
            capacity = self.capacity

        
        
        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "demand": demand / CAPACITIES[self.num_loc],        # normalize demands
                "stochastic_demand": stochastic_demand / CAPACITIES[self.num_loc],
                "weather": weather,
                "capacity": capacity,       # =1
            },
            batch_size=batch_size,
            device=self.device,
        )

    def generate_fixed_data(self, batch_size, ) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        batch_size_int = batch_size
        if not isinstance(batch_size, int):
            batch_size_int = batch_size[0]
        locs_with_depot = ACVRPEnv.a_loc_with_depot.repeat(batch_size_int, 1, 1)

        # Initialize the demand for nodes except the depot
        # Demand sampling Following Kool et al. (2019)
        # Generates a slightly different distribution than using torch.randint
        
        
        demand = ACVRPEnv.a_demand.repeat(batch_size_int, 1)

        # Initialize the weather
        weather = (
            torch.FloatTensor(*batch_size, 3)
            .uniform_(-1, 1)
        ).to(self.device)
        
        #  E(stochastic demand) = E(demand)
        if self.generate_method == "uniform":
            # print(f"generate data by uniform")
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        elif self.generate_method == "modelize":
            # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

            stochastic_demand = get_stoch_var(demand.to("cpu"),
                                                   locs_with_depot[..., 1:, :].to("cpu").clone(), 
                                                   weather[:, None, :].
                                                   repeat(1, self.num_loc, 1).to("cpu"),
                                                   None,
                                                   stoch_idx=ACVRPEnv.stoch_idx).squeeze(-1).float().to(self.device)

        # Support for heterogeneous capacity if provided
        if not isinstance(self.capacity, torch.Tensor):
            capacity = torch.full((*batch_size,), self.capacity, device=self.device)
        else:
            capacity = self.capacity

        
        
        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "demand": demand / CAPACITIES[self.num_loc],        # normalize demands
                "stochastic_demand": stochastic_demand / CAPACITIES[self.num_loc],
                "weather": weather,
                "capacity": capacity,       # =1
            },
            batch_size=batch_size,
            device=self.device,
        )
        
    
    
    def reset_stochastic_demand(self, td, alpha):
        '''
        td is state of env, after calla reset()
        alpha: [batch, 9, 1]
        '''
        
        # reset real demand from weather
        batch_size = td["demand"].size(0)
        if self.generate_method == "uniform":
            # print(f"generate data by uniform")
            stochastic_demand = (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            ).float().to(self.device)
        elif self.generate_method == "no_stoch":
            stochastic_demand = td["demand"].clone()
        elif self.generate_method == "modelize":
            # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

            locs_cust = td["locs"].clone()
            locs_cust = locs_cust[:, 1:, :]
            stochastic_demand = get_stoch_var(td["demand"].to("cpu"),
                                                   locs_cust.to("cpu"), 
                                                   td["weather"][:, None, :].
                                                   repeat(1, self.num_loc, 1).to("cpu"),
                                                   alpha[:, None, ...].to("cpu"), 
                                                   stoch_idx=ACVRPEnv.stoch_idx).squeeze(-1).float().to(self.device)

        td.set("real_demand", stochastic_demand)
        
        return td
    
    def reset_stochastic_var(self, td, adver_out):
        td = self.reset_stochastic_demand(td, adver_out)
        return td
    
    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["real_demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if this time depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td
    
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # cannot mask exceeding node in svrp
        # exceeds_cap = td["demand"][:, None, :] + td["used_capacity"][..., None] > 1.0

        # Nodes that cannot be visited are already visited
        mask_loc = td["visited"][..., 1:].to(torch.bool)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)
    
    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            
            td = self.generate_data(batch_size=batch_size)
                
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)

        # Create reset TensorDict
        real_demand = (
            td["stochastic_demand"] 
        )
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "weather": td["weather"],
                "demand": td["demand"], # observed demand
                "real_demand": real_demand,
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    
    @staticmethod
    def complete_penaltied_actions(actions, demands, scale_cap):
        '''
        actions: 1dim, length=padding后的长度(之前补全最长的)
        demands: 1dim, actions对应的数据需求。 
        scale_cap: 标量，代表车的最大容量， 非归一化
        '''
        # 补充，第0元素是 depot的需求， =-cap
        demands = torch.cat([torch.tensor([-scale_cap]), demands])
        
        used_cap = torch.zeros((1), device=actions.device)
        penaltied_actions = []
        for i in range(actions.size(0)):
            penaltied_actions.append(int(actions[i].cpu()))
            used_cap += demands[actions[i]]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            if used_cap > scale_cap + 1e-5:        # 1 dim
                # print("Used more than capacity")
                penaltied_actions.append(0)     # 不能满足需求，回到depot，插入0节点
                penaltied_actions.append(int(actions[i].cpu())) # 再返回当前节点
                used_cap = demands[actions[i]]       # 当前用过的容量，重新变为该节点的需求
        return torch.tensor(penaltied_actions, device="cpu")
                
    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, **kwargs):
        ''''
        绘制出action: agent选择的路径
        demand. 变化前后的
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm, colormaps

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
        scale = CAPACITIES.get(td["locs"].size(-2) - 1, 1)
        demands = td["demand"] * scale
        real_demands = td["real_demand"] * scale

        # svrp中需要对action惩罚的loc做0的插入，再绘图
        actions = ACVRPEnv.complete_penaltied_actions(actions, real_demands, scale)
        print("complete actions is ", actions)
        # add the depot at the first action and the end action
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

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
        # plot demand bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.01,   # width
                    3*demands[node_idx - 1] / (scale * 10),       # height
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
            
        # text demand
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] + move_x,
                locs[node_idx, 1] - 0.025,
                f"{demands[node_idx-1].item():.2f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(0),
            )
            
        
        # plot real demand bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                    0.005,
                    3*real_demands[node_idx - 1] / (scale * 10),
                    edgecolor=cm.Set2(1),
                    facecolor=cm.Set2(1),
                    fill=True,
                )
            )

        # text real demand
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0] - 0.085+ move_x,
                locs[node_idx, 1] - 0.025,
                f"{real_demands[node_idx-1].item():.2f} |",
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