"""Plot rewards vs. norms.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import gym
import numpy as np
import os
import sys
import time

import seaborn

import matplotlib.pyplot as plt

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

import tensorflow as tf


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

def rollout(env, agent, args, multiagent=False):
    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    for i in range(args.num_rollouts):
        vel = []
        state = env.reset()
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for _ in range(args.horizon):
            vehicles = env.unwrapped.k.vehicle
            speeds = vehicles.get_speed(vehicles.get_ids())

            # only include non-empty speeds
            if speeds:
                vel.append(np.mean(speeds))

            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(
                            state[agent_id], policy_id=policy_map_fn(agent_id))
            else:
                action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))

    print(f'==== Finished epoch ====')
    return np.mean(np.array([np.mean(rew) for _, rew in rets.items()])) if multiagent else np.mean(rets)

def plot(args, l_1, l_2, lc, rew, e):
    color = seaborn.color_palette(palette="crest", as_cmap=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.set_title("Max L1 norm")
    ax2.set_title("Max L2 norm")
    ax3.set_title("Max Lipschitz Constant")
    for i in range(len(e)):
        if np.isnan(rew[i]):
            continue
        c = color(int(e[i])/5000)
        ax1.scatter(l_1[i], rew[i], c=[c])
        ax2.scatter(l_2[i], rew[i], c=[c])
        ax3.scatter(lc[i], rew[i], c=[c])

    #for i, txt in enumerate(epochs):
    #    ax1.annotate(txt, (l_1[i], rew[i]))
    #    ax2.annotate(txt, (l_2[i], rew[i]))

    fig.suptitle(f'Eta value of {args.results}. True value of 20')
    plt.savefig(args.save_path)
    plt.show()

def compute_norms(args):
    results = args.results if args.results[-1] != '/' \
        else args.results[:-1]

    l_1 = []
    l_2 = []
    lc = []
    rew = []
    e = []
    epochs = [str(e) for e in range(args.low, args.high+1, args.skip)]

    for result_dir in open(results+".txt", 'r'):
        result_dir = result_dir[:-1].rstrip() if result_dir[-1] == '/' else result_dir.rstrip()

        try:
            config = get_rllib_config(result_dir)
        except:
            continue

        # check if we have a multiagent environment but in a
        # backwards compatible way
        if config.get('multiagent', {}).get('policies', None):
            multiagent = True
            pkl = get_rllib_pkl(result_dir)
            config['multiagent'] = pkl['multiagent']
        else:
            multiagent = False

        # Run on only one cpu for rendering purposes
        config['num_workers'] = 0

        flow_params = get_flow_params(config)

        # hack for old pkl files
        # TODO(ev) remove eventually
        sim_params = flow_params['sim']
        setattr(sim_params, 'num_clients', 1)

        # for hacks for old pkl files TODO: remove eventually
        if not hasattr(sim_params, 'use_ballistic'):
            sim_params.use_ballistic = False

        # Determine agent and checkpoint
        config_run = config['env_config']['run'] if 'run' in config['env_config'] \
            else None
        if args.run and config_run:
            if args.run != config_run:
                print('visualizer_rllib.py: error: run argument '
                      + '\'{}\' passed in '.format(args.run)
                      + 'differs from the one stored in params.json '
                      + '\'{}\''.format(config_run))
                sys.exit(1)
        if args.run:
            agent_cls = get_agent_class(args.run)
        elif config_run:
            agent_cls = get_agent_class(config_run)
        else:
            print('visualizer_rllib.py: error: could not find flow parameter '
                  '\'run\' in params.json, '
                  'add argument --run to provide the algorithm or model used '
                  'to train the results\n e.g. '
                  'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
            sys.exit(1)

        sim_params.restart_instance = True
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Create and register a gym+rllib env
        create_env, env_name = make_create_env(params=flow_params, reward_specification=[('desired_vel', 1), ('accel', 20)])
        register_env(env_name, create_env)

        # Start the environment with the gui turned on and a path for the
        # emission file
        env_params = flow_params['env']
        env_params.restart_instance = False

        # lower the horizon if testing
        if args.horizon:
            config['horizon'] = args.horizon
            env_params.horizon = args.horizon

        # create the agent that will be used to compute the actions
        agent = agent_cls(env=env_name, config=config)

        if hasattr(agent, "local_evaluator") and \
                os.environ.get("TEST_FLAG") != 'True':
            env = agent.local_evaluator.env
        else:
            env = gym.make(env_name)

        # if restart_instance, don't restart here because env.reset will restart later
        if not sim_params.restart_instance:
            env.restart_simulation(sim_params=sim_params, render=sim_params.render)

        
        for epoch in epochs:
            checkpoint = result_dir + '/checkpoint_' + epoch
            checkpoint = checkpoint + '/checkpoint-' + epoch
            if not os.path.isfile(checkpoint):
                break
            agent.restore(checkpoint)

            weights = [w for _, w in agent.get_weights()['default_policy'].items()]
            sv = np.array([np.linalg.svd(w, compute_uv=False)[0] for w in weights[::4]])
            kernel_norm1 = [np.linalg.norm(w, ord=1) for w in weights[::4]]
            kernel_norm2 = [np.linalg.norm(w, ord=2) for w in weights[::4]]
            bias_norm1 = [np.linalg.norm(w, ord=1) for w in weights[1::4]]
            bias_norm2 = [np.linalg.norm(w, ord=2) for w in weights[1::4]]

            l_1.append(max(np.max(kernel_norm1), np.max(bias_norm1)))
            l_2.append(max(np.max(kernel_norm2), np.max(bias_norm2)))
            lc.append(np.prod(sv))
            rew.append(rollout(env, agent, args, multiagent=multiagent))
            e.append(epoch)

        # terminate the environment
        env.unwrapped.terminate()

    print(l_1)
    print(l_2)
    print(lc)
    print(rew)
    print(e)

    plot(args, l_1, l_2, lc, rew, e)
    

    

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'results', type=str, help='File with list of directory containing results')
    #parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=3,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--horizon',
        default=300,
        type=int,
        help='Specifies the horizon.')
    parser.add_argument('--low', type=int, default=500, help='the epoch to start plotting from')
    parser.add_argument('--high', type=int, default=5000, help='the epoch to stop plotting from')
    parser.add_argument('--skip', type=int, default=500, help='the epoch to stop plotting at')
    parser.add_argument('--save_path', type=str, default="f.png", help="savepath of figure")

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=0)
    compute_norms(args)
