import argparse
import logging
import os
import sys

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger
import gc
log = get_pylogger(__name__)


DISTRIBUTIONS_PER_PROBLEM = {
    "acsp": [None],
    "vrp": [None],
    "acvrp": ["modelize", "uniform", "no_stoch"],
    "pg": ["const", "unif", "dist"],
}

from rl4co.envs.routing.env_utils import STOCH_PARAMS
STOCH_IDX = 0

def generate_env_data(env_type, *args, **kwargs):
    """Generate data for a given environment type in the form of a dictionary"""
    try:
        # breakpoint()
        # remove all None values from args
        args = [arg for arg in args if arg is not None]

        return getattr(sys.modules[__name__], f"generate_{env_type}_data")(
            *args, **kwargs
        )
    except AttributeError:
        raise NotImplementedError(f"Environment type {env_type} not implemented")


def generate_acsp_data(dataset_size, csp_size):
    cover_low = 0.
    cover_high = 0.25
    locs = np.random.uniform(size=(dataset_size, csp_size, 2)).astype(np.float32)
    size = csp_size
    max_cover = np.random.uniform(size=(dataset_size, csp_size), low=cover_low, high=cover_high).astype(np.float32)
    weather = np.random.uniform(low=-1., high=1., size=(dataset_size, 3)).astype(np.float32)
    stochastic_maxcover = get_stoch_var(max_cover[..., np.newaxis],
                                          locs.copy(),
                                                np.repeat(weather[:, np.newaxis, :], size, axis=1),
                                                None).squeeze(-1).astype(np.float32)
    stochastic_maxcover = np.clip(stochastic_maxcover, a_min=cover_low, a_max=cover_high)
    min_cover = np.ones_like(max_cover) * cover_low
    return {
        "locs": locs,
        "max_cover": max_cover,
        "weather": weather,
        "stochastic_maxcover": stochastic_maxcover,
        "min_cover": min_cover,
    }

def generate_tsp_data(dataset_size, tsp_size):
    return {
        "locs": np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)
    }

    
def generate_vrp_data(dataset_size, vrp_size, capacities=None):
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

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v

    return {
        "depot": np.random.uniform(size=(dataset_size, 2)).astype(
            np.float32
        ),  # Depot location
        "locs": np.random.uniform(size=(dataset_size, vrp_size, 2)).astype(
            np.float32
        ),  # Node locations
        "demand": np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
            np.float32
        ),  # Demand, uniform integer 1 ... 9
        "capacity": np.full(dataset_size, CAPACITIES[vrp_size]).astype(np.float32),
    }  # Capacity, same for whole dataset
# @profile(stream=open('logmem_generate_svrp_data.log', 'w+'))
def generate_acvrp_data(dataset_size, vrp_size, generate_type="modelize", capacities=None):
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

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v
    
    demand = np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
            np.float32
        )  # Demand, uniform integer 1 ... 9
    
    weather = np.random.uniform(low=-1., high=1., size=(dataset_size, 3)).astype(
            np.float32
        )  # Weather, uniform float (-1, 1)
    
    locs = np.random.uniform(size=(dataset_size, vrp_size, 2)).astype(
            np.float32
        )
    #  E(stochastic demand) = E(demand)
    if generate_type == "uniform":
        stochastic_demand = np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
            np.float32
        )
    elif generate_type == "no_stoch":
        stochastic_demand = demand.copy()
    elif generate_type == "modelize":
        # alphas = torch.rand((n_problems, n_nodes, 9, 1))      # =np.random.random, uniform dis(0, 1)

        stochastic_demand = get_stoch_var(demand[..., np.newaxis],
                                          locs.copy(),
                                                np.repeat(weather[:, np.newaxis, :], vrp_size, axis=1),
                                                None).squeeze(-1).astype(np.float32)
            
    return {
        "depot": np.random.uniform(size=(dataset_size, 2)).astype(
            np.float32
        ),  # Depot location
        "locs": locs,  # Node locations
        "demand": demand,
        "stochastic_demand": stochastic_demand,
        "weather": weather,
        "capacity": np.full(dataset_size, CAPACITIES[vrp_size]).astype(np.float32),
    }  # Capacity, same for whole dataset


def get_stoch_var(inp, locs, w, alphas, A=0.6, B=0.2, G=0.2):
    A, B, G = STOCH_PARAMS[STOCH_IDX]   # 和env的stoch_idx一致
    print(f"ABG in generate data is {A} {B} {G}")
    n_problems,n_nodes,shape = inp.shape
    T = inp/A
    
    var_noise = T*G
    # noise = np.random.randn(n_problems,n_nodes, shape)      #=np.rand.randn, normal dis(0, 1)
    # noise = var_noise*noise     # multivariable normal distr, var_noise mean
    # noise = np.clip(noise, a_min=-var_noise, a_max=var_noise)
    
    noise = np.sqrt(var_noise)*np.random.randn(n_problems,n_nodes, shape)      #=np.rand.randn, normal dis(0, 1)
    noise = np.clip(noise, a_min=-var_noise, a_max=var_noise)
    
    # var_w = np.sqrt(T*B)
    var_w = T*B
    # sum_alpha = var_w[:, :, np.newaxis, :]*4.5      #? 4.5
    # alphas = np.random.random((n_problems, n_nodes, 9, shape))      # =np.random.random, uniform dis(0, 1)
    # alphas /= alphas.sum(axis=2)[:, :, np.newaxis, :]       # normalize alpha to 0-1
    # alphas *= sum_alpha     # alpha value [4.5*var_w]
    # alphas = np.sqrt(alphas)        # alpha value [sqrt(4.5*var_w)]
    # signs = np.random.random((n_problems, n_nodes, 9, shape))
    # signs = np.where(signs > 0.5)
    # alphas[signs] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
    
    # sum_alpha = var_w[:, :, None, :]*4.5      #? 4.5
    sum_alpha = var_w[:, :, np.newaxis, :]*9      #? 4.5
    # alphas = np.random.random((n_problems, n_nodes, 9, shape))      # =np.random.random, uniform dis(0, 1)
    alphas = np.random.random((n_problems, 1, 9, shape))      # =np.random.random, uniform dis(0, 1)
    alphas_loc = locs.sum(-1)[..., np.newaxis, np.newaxis]/2 * alphas
    
    
    # alphas_loc /= alphas_loc.sum(axis=2)[:, :, np.newaxis, :]       # normalize alpha to 0-1
    alphas_loc *= sum_alpha     # alpha value [4.5*var_w]
    # alphas_loc = np.sqrt(alphas_loc)        # alpha value [sqrt(4.5*var_w)]
    signs = np.random.random((n_problems, n_nodes, 9, shape))
    alphas_loc[np.where(signs > 0.5)] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
        
    # w1 = np.repeat(w, 3, axis=2)[..., np.newaxis]       # [batch, nodes, 3*repeat3=9, 1]
    # # roll shift num in axis: [batch, nodes, 3] -> concat [batch, nodes, 9,1]
    # w2 = np.concatenate([w, np.roll(w,shift=1,axis=2), np.roll(w,shift=2,axis=2)], 2)[..., np.newaxis]
    
    # tot_w = (alphas*w1*w2).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
    # tot_w = np.clip(tot_w, a_min=-var_w, a_max=var_w)
    
   
    tot_w = (alphas_loc*
             np.repeat(w, 3, axis=2)[..., np.newaxis]*
             np.concatenate([w, np.roll(w,shift=1,axis=2), 
                             np.roll(w,shift=2,axis=2)], 2)[..., np.newaxis]
            ).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
    tot_w = np.clip(tot_w, a_min=-var_w, a_max=var_w)
    out = inp + tot_w + noise
    out = np.clip(out, a_min=0.0, a_max=1e5)       # a_min a_max必须都是标量
    
    del sum_alpha, alphas_loc, signs, tot_w
    del T, noise, var_w
    # gc.collect()
        
    return out



def generate_pg_data(dataset_size, op_size, prize_type="const", max_lengths=None):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == "const":
        prize = np.ones((dataset_size, op_size))
    elif prize_type == "unif":
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.0
    else:  # Based on distance to depot
        assert prize_type == "dist"
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (
            1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)
        ) / 100.0

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
    max_lengths = MAX_LENGTHS if max_lengths is None else max_lengths

    return {
        "depot": depot.astype(np.float32),
        "locs": loc.astype(np.float32),
        "prize": prize.astype(np.float32),
        "max_length": np.full(dataset_size, max_lengths[op_size]).astype(np.float32),
    }


def generate_dataset(
    filename=None,
    data_dir="data",
    name=None,
    problem="all",
    data_distribution="all",
    dataset_size=10000,
    graph_sizes=[20, 50, 100],
    overwrite=False,        # if exists, skip regenerate
    seed=1234,
    disable_warning=True,
    distributions_per_problem=None,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """
    assert filename is None or (
        len(problem) == 1 and len(graph_sizes) == 1
    ), "Can only specify filename when generating a single dataset"

    distributions_per_problem = DISTRIBUTIONS_PER_PROBLEM

    if problem == "all":
        problems = distributions_per_problem
    else:
        problems = {
            problem: distributions_per_problem[problem]
            if data_distribution == "all"
            else [data_distribution]
        }
    # breakpoint()
    fname = filename
    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in graph_sizes:
                datadir = os.path.join(data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if filename is None:
                    fname = os.path.join(
                        datadir,
                        "{}{}{}_{}_seed{}_size{}.npz".format(
                            problem,
                            "_{}".format(distribution)
                            if distribution is not None
                            else "",
                            graph_size,
                            name,
                            seed,
                            dataset_size,
                        ),
                    )
                else:
                    fname = check_extension(filename, extension=".npz")

                if not overwrite and os.path.isfile(
                    check_extension(fname, extension=".npz")
                ):
                    if not disable_warning:
                        log.info(
                            "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                                fname
                            )
                        )
                    continue

                # Set seed
                np.random.seed(seed)

                # Automatically generate dataset
                dataset = generate_env_data(
                    problem, dataset_size, graph_size, distribution
                )

                # A function can return None in case of an error or a skip
                if dataset is not None:
                    # Save to disk as dict
                    log.info("Saving {} dataset to {}".format(problem, fname))
                    np.savez(fname, **dataset)
# @profile(stream=open('log_mem3_genedata.log', 'w+'))
def generate_default_datasets(data_dir, data_cfg):
    """Generate the default datasets used in the paper and save them to data_dir/problem"""
    # 传入大小的是，在测试时为了快速验证
    generate_dataset(data_dir=data_dir, dataset_size=data_cfg["val_data_size"], name="val", problem="all", seed=4321)
    generate_dataset(data_dir=data_dir, dataset_size=data_cfg["test_data_size"], name="test", problem="all", seed=1234)
    # 平时使用不传入大小的
    # generate_dataset(data_dir=data_dir, name="val", problem="all", seed=4321)
    # generate_dataset(data_dir=data_dir, name="test", problem="all", seed=1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", help="Filename of the dataset to create (ignores datadir)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Create datasets in data_dir/problem (default 'data')",
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name to identify dataset"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
        " or 'all' to generate all",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="all",
        help="Distributions to generate for problem, default 'all'.",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--disable_warning", action="store_true", help="Disable warning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.overwrite = args.f
    delattr(args, "f")
    generate_dataset(**vars(args))
