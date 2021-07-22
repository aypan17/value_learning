import importlib
import numpy as np

from flow.core import rewards


def local(env, rl_actions, **kwargs):
    # return a reward of 0 if a collision occurred
    if kwargs["fail"]:
        return 0

    # reward high system-level velocities
    if self.local_reward == 'local':
        rl_veh = self.rl_veh
        cost1 = rewards.local_desired_velocity(self, self.rl_veh, fail=kwargs["fail"])
    elif self.local_reward == 'partial_first':
        cost1 = rewards.local_desired_velocity(self, self.rl_veh[:3], fail=kwargs["fail"])
    elif self.local_reward == 'partial_last':
        cost1 = rewards.local_desired_velocity(self, self.rl_veh[-3:], fail=kwargs["fail"])
    else:
        cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

    # penalize small time headways
    cost2 = 0
    t_min = 1  # smallest acceptable time headway
    for rl_id in self.rl_veh:
        lead_id = self.k.vehicle.get_leader(rl_id)
        if lead_id not in ["", None] \
                and self.k.vehicle.get_speed(rl_id) > 0:
            t_headway = max(
                self.k.vehicle.get_headway(rl_id) /
                self.k.vehicle.get_speed(rl_id), 0)
            cost2 += min((t_headway - t_min) / t_min, 0)

    cost3 = 0
    mean_actions = np.mean(np.abs(np.array(rl_actions)))
    accel_threshold = 0

    if mean_actions > accel_threshold:
        cost3 += accel_threshold - mean_actions

    # weights for cost1, cost2, and cost3, respectively
    eta1, eta2, eta3 = 1.00, 0.10, self.eta

    return float(eta1 * cost1 + eta2 * cost2 + eta3 * cost3)


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
        self.reward_specification = reward_specification   
        assert all([r in rewards.REWARD_REGISTRY for r, _ in self.reward_specification])

        def proxy_reward(rl_actions, **kwargs):
            vel = np.array(self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
            if any(vel < -100) or kwargs["fail"]:
                return 0
            rew = 0 
            for name, eta in self.reward_specification:
                rew += eta * rewards.REWARD_REGISTRY[name](self.env, rl_actions)
            return rew 

        setattr(self.env, "compute_reward", proxy_reward)

    def __getattr__(self, attr):
        return self.env.__getattribute__(attr)

