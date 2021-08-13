from tqdm import trange

import torch

import time
import numpy as np

import pandemic_simulator as ps
from pandemic_simulator.environment.reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.callback import WandbCallback, SacdCallback
import sys
import wandb

import argparse

from sacd.agent import SacdAgent, SharedSacdAgent

GAMMA = float(sys.argv[10])

def make_cfg():
    #cfg =  ps.sh.small_town_config
    #cfg.delta_start_lo = float(sys.argv[7])
    #cfg.delta_start_hi = float(sys.argv[8])
    #return cfg
    sim_config = ps.env.PandemicSimConfig(
         num_persons=500,
         location_configs=[
             ps.env.LocationConfig(ps.env.Home, num=150),
             ps.env.LocationConfig(ps.env.GroceryStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)),
             ps.env.LocationConfig(ps.env.School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Hospital, num=1, num_assignees=15, state_opts=dict(patient_capacity=5)),
             ps.env.LocationConfig(ps.env.RetailStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.HairSalon, num=2, num_assignees=3, state_opts=dict(visitor_capacity=5)),
             ps.env.LocationConfig(ps.env.Restaurant, num=1, num_assignees=6, state_opts=dict(visitor_capacity=30)),
             ps.env.LocationConfig(ps.env.Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30))
         ],
         person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment(),
	 delta_start_lo = float(sys.argv[6]),
	 delta_start_hi = float(sys.argv[7])
    )
    return sim_config

def make_reg():
    return ps.sh.austin_regulations

def make_sim(sim_config):
    return ps.env.PandemicSim.from_config(sim_config)

def make_viz(sim_config):
    return ps.viz.GymViz.from_config(sim_config=sim_config)

def make_model(env):
    agent = ps.model.StageModel(env = env)

    # from torch.nn import Softsign, ReLU
    ppo_params = {'n_steps': 200, 
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

    d_model = int(sys.argv[8])
    n_layers = int(sys.argv[9])
    net_arch = [d_model] * n_layers if n_layers != 0 else []

    policy_kwargs = {
        "net_arch": [dict(pi=net_arch, vf=net_arch)], 
        #"activation_fn": torch.nn.ReLU
    }

    model = agent.get_model("ppo",  
                            model_kwargs = ppo_params, 
                            policy_kwargs = policy_kwargs, verbose = 0)

    return model

def init(args):
    cfg = wandb.config
    n_cpus = args.n_cpus
    ps.init_globals(seed=0)
    sim_config = make_cfg()
    regulations = make_reg()
    viz = make_viz(sim_config)
    done_fn = ps.env.DoneFunctionFactory.default(ps.env.DoneFunctionType.TIME_LIMIT, horizon=200)

    reward_fn = SumReward(
            reward_fns=[
#                RewardFunctionFactory.default(RewardFunctionType.ELDERLY_HOSPITALIZED),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3*sim_config.max_hospital_capacity),
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
            constrain=True,
            obs_history_size=24,
        )
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()
    return env, gym.get_single_env(), viz

def train(env, test_env, viz, args):
    model = make_model(env)
    print("Running model")
    if args.test:
        model.learn(total_timesteps = 2048, callback = WandbCallback(name=sys.argv[1], gamma=GAMMA, viz=viz, multiprocessing=(args.n_cpus>1)))
    else:
        model.learn(total_timesteps = 2048 * 500, callback = WandbCallback(name=sys.argv[1], gamma=GAMMA, viz=viz, multiprocessing=(args.n_cpus>1)))
    return model    

def train_sacd(env, test_env, viz, args):
    cfg = wandb.config
    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=args.log_dir, cuda=args.cuda,
        seed=args.seed, **cfg)
    agent.run(callback=SacdCallback(name=sys.argv[1], gamma=GAMMA))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--sacd', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default="pan_log")
    args = parser.parse_known_args(sys.argv[1:])[0]

    config = {
        'memory_size': 300000,
        'gamma': GAMMA,
        'multi_step': 1,
        'target_entropy_ratio': 0.98,
        'start_steps': 20000,
        'update_interval': 4,
        'target_update_interval': 8000,
        'use_per': False,
        'dueling_net': False,
        'num_steps': 300000,
        'num_eval_steps': 128,
        'max_episode_steps': 128,
        'log_interval': 1000000,
        'eval_interval': 10000000,
        'batch_size': 64,
        'd_model': 128, 
        'actor_lr': 0.0001, 
        'critic_lr': 0.001
    }
    if args.test:
        wandb.init(
          project="test-space",
          group="covid",
          entity="aypan17",
          config=config,
          sync_tensorboard=True
        )
    else:
        wandb.init(
          project="value-learning",
          group="covid",
          entity="aypan17",
          config=config,
          sync_tensorboard=True
        )
    if args.sacd:
        args.n_cpus=1
        train_env, test_env, viz = init(args)
        train_sacd(train_env, test_env, viz, args)
    else:
        train_env, test_env, viz = init(args)
        train(train_env, test_env, viz, args)


if __name__ == '__main__':
    main()
