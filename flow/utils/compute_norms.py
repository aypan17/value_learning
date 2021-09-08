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
from copy import deepcopy
import json

import seaborn
import scipy
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
from flow.core.rewards import REWARD_REGISTRY

import tensorflow as tf


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

class DiagGaussian(object):
    """Action distribution where each vector element is a gaussian.

    The first half of the input vector defines the gaussian means, and the
    second half the gaussian standard deviations.
    """

    def __init__(self, inputs):
        mean, log_std = np.split(inputs, 2)
        self.mean = mean
        self.log_std = log_std
        self.std = np.exp(log_std)

    def kl(self, other):
        if other is None:
            return 0
        assert isinstance(other, DiagGaussian)
        if other.mean.shape != self.mean.shape:
            return None
        return np.sum(
            other.log_std - self.log_std +
            (np.square(self.std) + np.square(self.mean - other.mean)) /
            (2.0 * np.square(other.std)))

    @property
    def entropy(self):
        return np.sum(
            self.log_std + .5 * np.log(2.0 * np.pi * np.e))


def rollout(env, agent, args, baseline_agent=None):
    # Simulate and collect metrics
    rets = []
    true_rets = []
    true_rets2 = []
    actions = []
    log_probs = []
    vfs = []
    base_vfs = []
    kls = []
    car_kls = []

    for i in range(args.num_rollouts):
        ret = 0
        true_ret = 0
        true_ret2 = 0
        action_moments = [] 
        log_prob = []
        base_log_prob = []
        vf = []
        base_vf = []
        kl = []
        car_kl = []

        state = env.reset()
        for j in range(args.horizon):
            action = agent.compute_action(state, full_fetch=True)
            if baseline_agent:
                baseline_action = baseline_agent.compute_action(state, full_fetch=True)

            vf_preds = action[2]['vf_preds']
            logp = action[2]['action_logp']
            logits = action[2]['behaviour_logits']
            if baseline_agent:
                base_vf_preds = baseline_action[2]['vf_preds']
                base_logp = baseline_action[2]['action_logp']
                base_logits = baseline_action[2]['behaviour_logits']
            cars = []
            car_logits = []
            car_base_logits = []
            for i, rl_id in enumerate(env.unwrapped.rl_veh):
                # get rl vehicles inside the network
                if rl_id in env.unwrapped.k.vehicle.get_rl_ids():
                    cars.append(i)
            for c in cars:
                car_logits.append(logits[c])
                if baseline_agent:
                    car_base_logits.append(base_logits[c])
            for c in cars:
                car_logits.append(logits[c + len(logits)//2])
                if baseline_agent:
                    car_base_logits.append(base_logits[c])
            car_logits = np.array(car_logits)
            car_base_logits = np.array(car_base_logits)
            action = action[0]

            if (j+1) % 10 == 0:
                vf.append(vf_preds)
                log_prob.append(logp)
                action_moments.append((np.mean(action).item(), np.std(action).item()))
                action_dist = DiagGaussian(logits)
                if baseline_agent:
                    base_log_prob.append(base_logp)
                    base_vf.append(base_vf_preds)
                    base_action_dist = DiagGaussian(base_logits)
                    kl.append(action_dist.kl(base_action_dist))
                    if len(cars) > 0:
                        car_action_dist = DiagGaussian(car_logits)
                        car_base_action_dist = DiagGaussian(car_base_logits)
                        car_kl.append(car_action_dist.kl(car_base_action_dist))

            state, reward, done, _ = env.step(action)
            ret += reward
            vels = np.array([env.unwrapped.k.vehicle.get_speed(veh_id) for veh_id in env.unwrapped.k.vehicle.get_ids()])
            if all(vels > -100):
                # true_ret += REWARD_REGISTRY['bus'](env, action)
                # true_ret += REWARD_REGISTRY['accel'](env, action)
                # true_ret += 0.1 * REWARD_REGISTRY['headway'](env, action)
                true_ret += REWARD_REGISTRY['vel'](env, action) 
                true_ret += 5 * REWARD_REGISTRY['accel'](env, action)
                true_ret2 += REWARD_REGISTRY['vel'](env, action)
                true_ret2 += REWARD_REGISTRY['accel'](env, action)

            if done:
                break

        rets.append(ret)
        true_rets.append(true_ret)
        true_rets2.append(true_ret2)
        actions.append(action_moments)
        base_log_probs.append(base_log_prob)
        log_probs.append(log_prob)
        vfs.append(vf)
        base_vfs.append(base_vf)
        kls.append(kl)
        car_kls.append(car_kl)

        # outflow = vehicles.get_outflow_rate(500)
        # final_outflows.append(outflow)
        # inflow = vehicles.get_inflow_rate(500)
        # final_inflows.append(inflow)
        # if np.all(np.array(final_inflows) > 1e-5):
        #     throughput_efficiency = [x / y for x, y in
        #                              zip(final_outflows, final_inflows)]
        # else:
        #     throughput_efficiency = [0] * len(final_inflows)
        # mean_speed.append(np.mean(vel))
        # std_speed.append(np.std(vel))

    print(f'==== Finished epoch ====')
    if baseline_agent:
        base_log_probs, base_vfs, kls, car_kls = np.mean(base_log_probs, axis=0), np.mean(base_vfs, axis=0), np.mean(kls, axis=0), np.mean(car_kls, axis=0)
    else:
        base_log_probs, base_vfs, kls, car_kls = None, None, None, None
    return np.mean(rets), np.mean(true_rets), np.mean(true_rets2), actions, np.mean(log_probs, axis=0), base_log_probs, np.mean(vfs, axis=0), base_vfs, kls, car_kls

def plot(args, l_1, l_2, lc, p2r, rew, e):
    color = seaborn.color_palette(palette="crest", as_cmap=True)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
    ax1.set_title("Max L1 norm")
    ax2.set_title("Max L2 norm")
    ax3.set_title("Max Lipschitz Constant")
    ax4.set_title("Min reward vs. params")
    ax5.set_title("Last reward vs. params")
    for i in range(len(e)):
        if np.isnan(rew[i]) or np.isnan(lc[i]):
            continue
        c = color(int(e[i])/5000)
        ax1.scatter(l_1[i], rew[i], c=[c])
        ax2.scatter(l_2[i], rew[i], c=[c])
        ax3.scatter(lc[i], rew[i], c=[c])

    c = color(1)
    for p, (min_r, fin_r) in p2r.items():
        ax4.scatter(p, min_r, c='blue')
        ax5.scatter(p, fin_r, c='blue')

    name = args.results.split("/")[-1]
    fig.suptitle(f'{name}')
    plt.savefig(name+".png")
    plt.show()

def compute_norms(args):
    results = args.results if args.results[-1] != '/' \
        else args.results[:-1]

    params = []
    l_1 = []
    l_2 = []
    lc = []
    rew = []
    true_rew = []
    true_rew2 = []
    actions = []
    log_probs = []
    base_log_probs = []
    vfs = []
    base_vfs = []
    kls = []
    car_kls = []
    e = []
    m = []
    epochs = [str(i) for i in range(args.low, args.high+1, args.skip)]
    print(epochs)

    for directory in os.listdir(results):
        # misspecification = float(directory.split("_")[-1])
        # print(misspecification)
        for d in os.listdir(results+'/'+directory):
            result_dir = results + '/' + directory + '/' + d
            print(result_dir)
            try:
                config = get_rllib_config(result_dir)
            except:
                print(f"Loading {result_dir} config failed")
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
            sim_params = flow_params['sim']
            setattr(sim_params, 'num_clients', 1)

            # for hacks for old pkl files 
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
            merge=[('desired_vel', 1), ('accel', 20)]
            merge_first=[('partial_first', 1), ('headway', 0.1), ('accel', 1)]
            bus=[('vel', 1), ('headway', 0.1), ('accel', 1)]
            bottle=[('desired_vel', 1), ('forward', 0.1), ('lane_change_bool', 1)]
            outflow=[('outflow', 1)]
            create_env, env_name = make_create_env(params=flow_params, reward_specification=merge_first)
            register_env(env_name, create_env)
            create_env2, env_name2 = make_create_env(params=flow_params, reward_specification=merge_first)
            register_env(env_name2, create_env2)

            # Start the environment with the gui turned on and a path for the
            # emission file
            env_params = flow_params['env']
            env_params.restart_instance = False

            # lower the horizon if testing
            if args.horizon:
                config['horizon'] = args.horizon
                env_params.horizon = args.horizon

            # create the agent that will be used to compute the actions
            del config['callbacks']

            agent = agent_cls(env=env_name, config=config)
            if args.baseline:
                baseline_agent = agent_cls(env=env_name2, config=config)
                checkpoint = result_dir + '/checkpoint_' + epochs[0]
                checkpoint = checkpoint + '/checkpoint-' + epochs[0]
                baseline_agent.restore(checkpoint)
            else:
                baseline_agent = None

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

                r, tr, tr2, a, logp, base_logp, vf, base_vf, kl, car_kl = rollout(env, agent, args, baseline_agent=baseline_agent)
                weights = [w for _, w in agent.get_weights()['default_policy'].items()]
                names = [k for k, _ in agent.get_weights()['default_policy'].items()]

                try:
                    sv = np.array([scipy.linalg.svd(w, compute_uv=False, lapack_driver='gesvd')[0] for w in weights[::4]])
                    kernel_norm1 = [np.linalg.norm(w, ord=1) for w in weights[::4]]
                    kernel_norm2 = [np.linalg.norm(w, ord=2) for w in weights[::4]]
                    bias_norm1 = [np.linalg.norm(w, ord=1) for w in weights[1::4]]
                    bias_norm2 = [np.linalg.norm(w, ord=2) for w in weights[1::4]]

                    params.append(np.sum([np.prod(w.shape) for w in weights[::4]]).item())
                    l_1.append(float(max(np.max(kernel_norm1), np.max(bias_norm1))))
                    l_2.append(float(max(np.max(kernel_norm2), np.max(bias_norm2))))
                    lc.append(np.prod(sv).item())
                    
                    rew.append(r)
                    true_rew.append(tr)
                    true_rew2.append(tr2)
                    actions.append(a)
                    log_probs.append(logp.tolist())
                    vfs.append(vf.tolist())
                    if args.baseline:
                        base_log_probs.append(base_logp.tolist())
                        base_vfs.append(vf.tolist())
                        kls.append(kl.tolist())
                        car_kls.append(car_kl.tolist())
                    e.append(epoch)
                except:
                    continue
                #m.append(misspecification)

            # terminate the environment
            env.unwrapped.terminate()

    # p2r = {}
    # for p, r in zip(params, rew):
    #     if p not in p2r:
    #         p2r[p] = (r, r)
    #     else:
    #         p2r[p] = (min(p2r[p][0], r), r)

    print(m)
    print(params)
    print(l_1)
    print(l_2)
    print(lc)
    print(rew)
    print(true_rew)
    print(true_rew2)
    print(actions)
    print(log_probs)
    print(base_log_probs)
    print(vfs)
    print(base_vfs)
    print(kls)
    print(car_kls)
    print(e)
    #print(p2r)
    with open(f'{results}.json', 'a', encoding='utf-8') as f:
        json.dump({'m': m, 'params':params, 'l_1': l_1, 'l_2': l_2, 'lc': lc, 'rew': rew, 'true_rew': true_rew, 'true_rew2': true_rew2,
            'actions': actions, 'log_probs': log_probs, 'base_log_probs': base_log_probs, 'vfs': vfs, 'base_vfs': base_vfs, 'kls': kls, 'car_kls': car_kls, 'e': e}, f)
    f.close()
    # f = open(f"{results}.txt", "a")
    # f.write(str(m)+"\n")
    # f.write(str(params)+"\n")
    # f.write(str(l_1)+"\n")
    # f.write(str(l_2)+"\n")
    # f.write(str(lc)+"\n")
    # f.write(str(rew)+"\n")
    # f.write(str(true_rew)+"\n")
    # f.write(str(actions)+"\n")
    # f.write(str(e)+"\n")
    # #f.write(str(p2r)+"\n")
    # f.close()

    #plot(args, l_1, l_2, lc, p2r, rew, e)
       

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
        default=4,
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
    parser.add_argument('--baseline', action='store_true', default=False, help="whether or not to use a baseline model for epochs")

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=0)
    compute_norms(args)
