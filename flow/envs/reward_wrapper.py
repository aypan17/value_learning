import importlib
import numpy as np

from flow.core.rewards import REWARD_REGISTRY

class ProxyRewardEnv(object):
    """
    Wraps the given environment in a proxy reward wrapper function that changes the reward provided to the environment.

    Params
    ------
    env: the Flow environment object that will be wrapped with a proxy
    reward_specification: a dict of reward_pairs with str keys corresponding to reward type and float values corresponding to the weight of that reward. 
                            The proxy reward is a linear combination of all the reward functions specified. 
    *args: environment args
    **kwargs: envrionment kwargs
    """
    def __init__(self, module, name, env_params, sim_params, network, simulator, reward_specification):
        cls = getattr(importlib.import_module(module), name)
        self.env = cls(env_params, sim_params, network, simulator)
        self.reward_specification = []
        self.noise = 0
        for name, eta in reward_specification:
            if name == 'action_noise':
                assert self.noise == 0 
                self.noise = eta
            else:
                assert name in REWARD_REGISTRY 
                self.reward_specification.append((REWARD_REGISTRY[name], eta))

        def proxy_reward(rl_actions, **kwargs):
            vel = np.array(self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
            if any(vel < -100) or kwargs["fail"]:
                return 0
            rew = 0 
            for fn, eta in self.reward_specification:
                rew += eta * fn(self.env, rl_actions)
            return rew 

        setattr(self.env, "compute_reward", proxy_reward)

    def __getattr__(self, attr):
        return self.env.__getattribute__(attr)

    def _apply_rl_actions(self, rl_actions):
        self.env._apply_rl_actions(rl_actions + np.random.normal(scale=self.noise, size=len(rl_actions)))

