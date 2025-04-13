import numpy as np
import nashpy as nash
def eval_allgraph(rewards_graph, prog_strategy, adver_strategy):
    '''
    rewards_graph: numpy array, shape:(graph_nums, pro_policy_Num, adv_policy_Num)
    adver_strategy: numpy array, shape:(adv_policy_Num)
    prog_strategy: numpy array, shape:(pro_policy_Num)
    '''
    reward_prog_graph = np.matmul(rewards_graph, np.array(adver_strategy))
    reward_adver_graph = np.matmul(reward_prog_graph, np.array(prog_strategy))
    return reward_adver_graph

def eval_noadver(payoff, prog_strategy):
    '''
    no adver
    '''
    A = np.array(payoff)
    rps = nash.Game(A)
    result = rps[prog_strategy, 1]
    return result[0]

def eval_noadver_allgraph(rewards_graph, prog_strategy):
    '''
    rewards_graph: numpy array, shape:(graph_nums, pro_policy_Num)
    prog_strategy: numpy array, shape:(pro_policy_Num)
    '''
    reward_prog_graphs = np.matmul(rewards_graph, np.array(prog_strategy))
    return reward_prog_graphs