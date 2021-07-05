# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController
from flow.envs import BottleneckDesiredVelocityEnv
from flow.networks import BottleneckNetwork
from flow.core.experiment import Experiment

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

import json
import argparse
from copy import deepcopy
from time import strftime
from copy import deepcopy

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
import logging

# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10

def run_exp(flow_rate,
            scaling=1,
            disable_tb=True,
            disable_ramp_meter=True,
            n_crit=1000,
            feedback_coef=20):
    # Set up SUMO to render the results, take a time_step of 0.5 seconds per simulation step
    sim_params = SumoParams(
        sim_step=0.5,
        render=True,
        overtake_right=False,
        restart_instance=False)

    vehicles = VehicleParams()

    # Add a few vehicles to initialize the simulation. The vehicles have all lane changing enabled, 
    # which is mode 1621
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=25,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
        num_vehicles=1)

    vehicles.add(
        veh_id="followerstopper",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1 * SCALING)

    # These are additional params that configure the bottleneck experiment. They are explained in more
    # detail below.

    controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
    num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
    additional_env_params = {
        "target_velocity": 40,
        "disable_tb": True,
        "disable_ramp_metering": True,
        "controlled_segments": controlled_segments,
        "symmetric": False,
        "observed_segments": num_observed_segments,
        "reset_inflow": False,
        "lane_change_duration": 5,
        "max_accel": 3,
        "max_decel": 3,
        "inflow_range": [1000, 2000],
        "n_crit": n_crit,
        "feedback_coeff": feedback_coef
    }

    # Set up the experiment to run for 1000 time steps i.e. 500 seconds (1000 * 0.5)
    env_params = EnvParams(
        horizon=1000, additional_params=additional_env_params)

    # Add vehicle inflows at the front of the bottleneck. They enter with a flow_rate number of vehicles 
    # per hours and with a speed of 10 m/s
    flow_rate = 2300 * SCALING

    # percentage of flow coming out of each lane
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehs_per_hour=flow_rate * (1 - AV_FRAC),
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="followerstopper",
        edge="1",
        vehs_per_hour=flow_rate * AV_FRAC,
        departLane="random",
    departSpeed=10)

    # Initialize the traffic lights. The meanings of disable_tb and disable_ramp_meter are discussed later.
    traffic_lights = TrafficLightParams()
    if not disable_tb:
        traffic_lights.add(node_id="2")
    if not disable_ramp_meter:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": scaling, "speed_limit": 23}

    flow_params = dict(
        # name of the experiment
        exp_tag="DesiredVelocity",

        # name of the flow environment the experiment is running on
        env_name=BottleneckDesiredVelocityEnv,

        # name of the network class the experiment is running on
        network=BottleneckNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=False,
            print_warnings=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=40,
            sims_per_step=1,
            horizon=HORIZON,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="uniform",
            min_gap=5,
            lanes_distribution=float("inf"),
            edges_distribution=["2", "3", "4", "5"],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.params.TrafficLightParams)
        tls=traffic_lights,
    )


    # number of time steps
    flow_params['env'].horizon = 1000
    #exp = Experiment(flow_params)

    # run the sumo simulation
    #_ = exp.run(1)
    return flow_params

def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config

def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        '--multi', action='store_true', help='Run multiagent experiment')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]

if __name__ == '__main__':
    flow_params = run_exp(flow_rate=1000)
    alg_run, gym_name, config = setup_exps_rllib(flow_params, 2, 10)
    n_cpus=2
    ray.init(num_cpus=n_cpus + 1, object_store_memory=200 * 1024 * 1024)
    flags = parse_args(None)
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": flags.num_steps,
        },
    }

    if flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path
    run_experiments({flow_params["exp_tag"]: exp_config})