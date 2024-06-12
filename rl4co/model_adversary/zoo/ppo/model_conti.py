from rl4co.envs import RL4COEnvBase
from rl4co.model_adversary.rl import PPOContinuousAdversary
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.model_adversary.zoo.ppo.policy_conti import PPOContiAdvPolicy


class PPOContiAdvModel(PPOContinuousAdversary):
    """PPO Model based on Proximal Policy Optimization (PPO).

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        opponent: object or None,
        opponent_type: str,
        policy: PPOContiAdvPolicy = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = PPOContiAdvPolicy(env.name, **policy_kwargs)

        if critic is None:
            critic = CriticNetwork(env.name, **critic_kwargs)

        super().__init__(env, opponent, opponent_type, policy, critic, **kwargs)
