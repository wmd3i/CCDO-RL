import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPContext,
        "pg": pgContext,
        "acsp": TSPContext,
        "cvrp": VRPContext,
        "acvrp": ACVRPContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embedding_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embedding_dim = embedding_dim
        step_context_dim = (
            step_context_dim if step_context_dim is not None else embedding_dim
        )
        self.project_context = nn.Linear(
            step_context_dim, embedding_dim, bias=linear_bias
        )

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)

class pgContext(EnvContext):
    """Context embedding for the pg.
    Project the following to the embedding space:
        - first node embedding: depot
        - current node embedding
        - current time
    """

    def __init__(self, embedding_dim):
        super(pgContext, self).__init__(embedding_dim, 2 * embedding_dim + 1)
        # self.W_placeholder = nn.Parameter(
        #     torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        # )

    def _cur_tourtime_embedding(self, embeddings, td):
        """Get embedding of current tour time"""
        return td["tour_time"]
    def forward(self, embeddings, td):
        batch_size = [td.batch_size[0]]
        depot_idx = torch.ones((*batch_size, ), dtype=torch.int64).to(embeddings.device)
        depot_embedding = gather_by_index(embeddings, depot_idx)
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        tourtime_embedding = self._cur_tourtime_embedding(embeddings, td)
        context_embedding = torch.cat([depot_embedding, cur_node_embedding, tourtime_embedding[..., None]], -1)
        return self.project_context(context_embedding)
    
class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embedding_dim):
        super(TSPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)

class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embedding_dim):
        super(VRPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding

class ACVRPContext(EnvContext):
    """Context embedding for the Stochastic Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity) 
        - demand change of current nodes: demand - real demand
    """

    def __init__(self, embedding_dim):
        super(ACVRPContext, self).__init__(embedding_dim, embedding_dim + 1)
    
    def _state_embedding(self, embeddings, td):
        
        remain_capacity_embedding = td["vehicle_capacity"] - td["used_capacity"]
        
        is_depot = td["current_node"] == 0

        state_embedding = remain_capacity_embedding
        return state_embedding

