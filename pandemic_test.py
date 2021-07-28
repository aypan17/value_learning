from tqdm import trange

import pandemic_simulator as ps
#from model import StageModel
from pandemic_simulator.callback import WandbCallback
import sys
import wandb

def make_cfg():
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
        person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment()
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
    ppo_params ={'n_steps': 128, 
                 'ent_coef': 0.01, 
                 'learning_rate': 0.00005, 
                 'batch_size': 1024, 
                'gamma': 0.99}

    policy_kwargs = {
        "net_arch": [32, 32, 32], 
    }

    model = agent.get_model("ppo",  
                            model_kwargs = ppo_params, 
                            policy_kwargs = policy_kwargs, verbose = 0)

    return model

def train():
    cfg = wandb.config
    n_cpus = int(sys.argv[1])
    ps.init_globals(seed=0)
    # Loop
    # Make regulations
    # Run
    # Update gradients
    sim_config = make_cfg()
    regulations = make_reg()
    viz = make_viz(sim_config)
    true_viz = make_viz(sim_config)
    done_fn = ps.env.DoneFunctionFactory.default(ps.env.DoneFunctionType.TIME_LIMIT, horizon=128)
    gym = ps.env.PandemicPolicyGymEnv.from_config(
            sim_config=sim_config, 
            pandemic_regulations=regulations, 
            done_fn=done_fn,
            alpha=float(sys.argv[2]), 
            beta=float(sys.argv[3]), 
            gamma=float(sys.argv[4]), 
            delta=float(sys.argv[5]),
        )
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()

    model = make_model(env)
    print("Running model")
    model.learn(total_timesteps = 1024 * 200, callback = WandbCallback(viz=viz, true_viz=true_viz, multiprocessing=(n_cpus>1)))
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
