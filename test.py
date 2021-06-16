from flow.networks import RingNetwork
from flow.core.params import NetParams, InitialConfig, VehicleParams, SumoParams, EnvParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.envs import WaveAttenuationPOEnv

import json
import numpy as np

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from ray.tune.logger import Logger
from ray.tune.logger import DEFAULT_LOGGERS


# Callbacks
# Custom state can be stored for the episode in the info["episode"].user_data dict
# Custom scalar metrics reported by saving values to the info["episode"].custom_metrics dict
def on_episode_start(info):
    #print(info.keys())  # -> "env", 'episode", 'policy'
    episode = info["episode"]
    #env = info["env"]
    #print(episode.last)
    #print(info['env'])
    #print("episode {} started".format(episode.episode_id))
    episode.user_data["true_reward"] = []

def on_episode_step(info):

    episode = info["episode"]
    env = info["env"]

    kernel = env.vector_env.envs[0].k
    vel = np.array([
            kernel.vehicle.get_speed(veh_id)
            for veh_id in kernel.vehicle.get_ids()
        ])

    # reward average velocity
    eta_2 = 4.
    true_reward = eta_2 * np.mean(vel) / 20

    # punish accelerations (should lead to reduced stop-and-go waves)
    eta = 4  # 0.25
    mean_actions = np.mean(np.abs(np.array(episode.last_action_for())))
    accel_threshold = 0

    if mean_actions > accel_threshold:
        true_reward += eta * (accel_threshold - mean_actions)
    episode.user_data["true_reward"].append(true_reward)

def on_episode_end(info):
    
    episode = info["episode"]
    mean_rew = np.mean(episode.user_data["true_reward"])
    #print("episode {} ended with length {} and pole angles {}".format(
    #    episode.episode_id, episode.length, pole_angle))
    episode.custom_metrics["true_reward"] = mean_rew

def on_train_result(info):
    pass
    #print(info.keys())
    #pass
    #print("trainer.train() result: {} -> {} episodes".format(
    #    info["trainer"].__name__, info["result"]["episodes_this_iter"]))

def on_postprocess_traj(info):
    pass
    #episode = info["episode"]
    #batch = info["post_batch"]  # note: you can mutate this
    #print("postprocessed {} steps".format(batch.count))



def main():
    name = "misspecified_0.4"
    network_name = RingNetwork
    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    initial_config = InitialConfig(spacing="uniform", perturbation=1)

    vehicles = VehicleParams()
    vehicles.add("human",
                    acceleration_controller=(IDMController, {}),
                    routing_controller=(ContinuousRouter, {}),
                    num_vehicles=21)

    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)

    sim_params = SumoParams(sim_step=0.1, render=False)

    HORIZON=100

    env_params = EnvParams(
        horizon=HORIZON,

        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [320, 370]
        },
        )

    env_name = WaveAttenuationPOEnv

    # Creating flow_params. Make sure the dictionary keys are as specified. 
    flow_params = dict(
        # name of the experiment
        exp_tag=name,
        # name of the flow environment the experiment is running on
        env_name=env_name,
        # name of the network class the experiment uses
        network=network_name,
        # simulator that is used by the experiment
        simulator='traci',
        # simulation-related parameters
        sim=sim_params,
        # environment related parameters (see flow.core.params.EnvParams)
        env=env_params,
        # network-related parameters (see flow.core.params.NetParams and
        # the network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,
        # vehicles to be placed in the network at the start of a rollout 
        # (see flow.core.vehicles.Vehicles)
        veh=vehicles,
        # (optional) parameters affecting the positioning of vehicles upon 
        # initialization/reset (see flow.core.params.InitialConfig)
        initial=initial_config
    )

    # number of parallel workers
    N_CPUS = 2
    # number of rollouts per training iteration
    N_ROLLOUTS = 1

    ray.init(num_cpus=N_CPUS)

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS - 1  # number of parallel workers
    config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
    config["use_gae"] = True  # using generalized advantage estimation
    config["lambda"] = 0.97  
    config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
    config["kl_target"] = 0.02  # target KL divergence
    config["num_sgd_iter"] = 10  # number of SGD iterations
    config["horizon"] = HORIZON  # rollout horizon

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)  # generating a string version of flow_params
    config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
    config['env_config']['run'] = alg_run

    # Add callbacks
    config['callbacks'] = {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    "on_train_result": on_train_result,
                    "on_postprocess_traj": on_postprocess_traj,
                }

    # Call the utility function make_create_env to be able to 
    # register the Flow env for this experiment
    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env with Gym
    register_env(gym_name, create_env)


    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1,  # number of iterations between checkpoints
            "checkpoint_at_end": True,  # generate a checkpoint at the end
            "max_failures": 999,
            "stop": {  # stopping conditions
                "training_iteration": 250,  # number of iterations to stop after
            },
        },
        
    })

if __name__ == '__main__':
    main()