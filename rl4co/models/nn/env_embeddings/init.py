import torch
import torch.nn as nn


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPInitEmbedding,
        "pg": pgInitEmbedding,
        "acsp": ACSPInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "acvrp": ACVRPInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class ACSPInitEmbedding(nn.Module):
    """Initial embedding for the Covering Salesman Problems (CSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
        - min_cover
        - max_cover
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(ACSPInitEmbedding, self).__init__()
        node_dim = 3  # x, y,max_cover
        weather_dim = 3     # 
        self.init_embed = nn.Linear(node_dim+weather_dim, embedding_dim, linear_bias)

    def forward(self, td):
        size = td["locs"].shape[-2]
        weather = td["weather"][:, None, :].repeat(1, size, 1)
        feature = torch.concat([td["locs"],
                                #  td["min_cover"][..., None],
                                 td["max_cover"][..., None],
                                 weather], dim=-1)
        out = self.init_embed(feature)
        return out
    


class pgInitEmbedding(nn.Module):
    """Initial embedding for the pg
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
        - prize:
        - time window, low, high
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(pgInitEmbedding, self).__init__()
        weather_dim = 3
        node_dim = 10  # x, y, cost, attack_prob, tw_low, tw_high, width,  maxtime, prize/width, prize/todepot
        self.init_embed = nn.Linear(node_dim+weather_dim, embedding_dim, linear_bias)

    def forward(self, td):
        cost = td["cost"][..., None]
        prob = td["attack_prob"][..., None]
        tw_low = td["tw_low"][..., None]
        tw_high = td["tw_high"][..., None]
        width = tw_high - tw_low
        batch_size = td["cost"].shape[0]
        prize_dis_depot = td["adj"][range(batch_size), :, 0][..., None]
        size = cost.shape[-2]
        weather = td["weather"][:, None, :].repeat(1, size, 1)
        feature = torch.concat([td["locs"], prob, cost, tw_low, tw_high, width,
                                 td["maxtime"][..., None],
                                 cost / width,
                                 prize_dis_depot,
                                 weather], dim=-1)
        out = self.init_embed(feature)
        return out
    
class VRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(VRPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, demand
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)
        self.init_embed_depot = nn.Linear(
            2, embedding_dim, linear_bias
        )  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embedding_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embedding_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embedding_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out

class ACVRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(ACVRPInitEmbedding, self).__init__()
        weather_dim = 3
        node_dim = 3  # x, y, demand
        self.init_embed = nn.Linear(node_dim+weather_dim, embedding_dim, linear_bias)
        self.init_embed_depot = nn.Linear(
            2+weather_dim, embedding_dim, linear_bias
        )  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]

        # [batch, 1, 2]-> [batch, 1, 5]
        depot_vec = torch.cat((depot, td["weather"][:, None, :]), -1)    
        # [batch, n_city, 2]-> [batch, n_city, 5]
        cities_vec = torch.cat((cities, td["weather"][:, None, :].repeat(1, cities.size(1), 1)), -1)
        
        # [batch, 1, 5]-> [batch, 1, embedding_dim]
        depot_embedding = self.init_embed_depot(depot_vec)
        # [batch, n_city, 2]  -> [batch, n_city, embedding_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities_vec, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embedding_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out
    
