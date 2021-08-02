from tqdm import trange

import torch

import time
import numpy as np

import pandemic_simulator as ps
from pandemic_simulator.environment.reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.callback import WandbCallback
import sys
import wandb

GAMMA = float(sys.argv[5])

def make_cfg():
    return ps.sh.small_town_config
    # sim_config = ps.env.PandemicSimConfig(
    #     num_persons=500,
    #     location_configs=[
    #         ps.env.LocationConfig(ps.env.Home, num=150),
    #         ps.env.LocationConfig(ps.env.GroceryStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
    #         ps.env.LocationConfig(ps.env.Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)),
    #         ps.env.LocationConfig(ps.env.School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)),
    #         ps.env.LocationConfig(ps.env.Hospital, num=1, num_assignees=15, state_opts=dict(patient_capacity=5)),
    #         ps.env.LocationConfig(ps.env.RetailStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
    #         ps.env.LocationConfig(ps.env.HairSalon, num=2, num_assignees=3, state_opts=dict(visitor_capacity=5)),
    #         ps.env.LocationConfig(ps.env.Restaurant, num=1, num_assignees=6, state_opts=dict(visitor_capacity=30)),
    #         ps.env.LocationConfig(ps.env.Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30))
    #     ],
    #     person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment()
    # )
    # return sim_config

def make_reg():
    return ps.sh.austin_regulations

def make_sim(sim_config):
    return ps.env.PandemicSim.from_config(sim_config)

def make_viz(sim_config):
    return ps.viz.GymViz.from_config(sim_config=sim_config)

def make_model(env):
    agent = ps.model.StageModel(env = env)

    # from torch.nn import Softsign, ReLU
    ppo_params = {'n_steps': 128, 
                 'ent_coef': 0.01, 
                 'learning_rate': 0.0001, 
                 'batch_size': 64,  
                'gamma': GAMMA}

    sac_params = {
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.01",
        "gamma": GAMMA
    }

    d_model = int(sys.argv[7])
    n_layers = int(sys.argv[8])
    net_arch = [d_model] * n_layers if n_layers != 0 else []

    policy_kwargs = {
        "net_arch": [dict(pi=net_arch, vf=net_arch)], 
        #"activation_fn": torch.nn.ReLU
    }

    model = agent.get_model("ppo",  
                            model_kwargs = ppo_params, 
                            policy_kwargs = policy_kwargs, verbose = 0)

    return model

def train():
    cfg = wandb.config
    n_cpus = int(sys.argv[9])
    ps.init_globals(seed=0)
    # Loop
    # Make regulations
    # Run
    # Update gradients
    sim_config = make_cfg()
    regulations = make_reg()
    viz = make_viz(sim_config)
    done_fn = ps.env.DoneFunctionFactory.default(ps.env.DoneFunctionType.TIME_LIMIT, horizon=128)

    reward_fn = SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.ELDERLY_HOSPITALIZED),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(regulations))
            ],
            weights=[float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
        )

    gym = ps.env.PandemicPolicyGymEnv.from_config(
            sim_config=sim_config, 
            pandemic_regulations=regulations, 
            done_fn=done_fn,
            reward_fn=reward_fn,
            constrain=True
        )
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()

    model = make_model(env)
    print("Running model")
    model.learn(total_timesteps = 2048 * 500, callback = WandbCallback(name=sys.argv[1], viz=viz, multiprocessing=(n_cpus>1)))
    return model

def main():

    config = {}

    wandb.init(
      project="value-learning",
      group="covid",
      entity="aypan17",
      config=config,
      sync_tensorboard=True
    )
    train()


if __name__ == '__main__':
    main()
