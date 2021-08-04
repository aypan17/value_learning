import gym
from gym.envs.registration import register
from simglucose.callback import WandbCallback


def reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1

def make_model(env):
    agent = simglucose.model.BasalInsulinModel(env = env)

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

    d_model = 256 #int(sys.argv[7])
    n_layers = 1 #int(sys.argv[8])
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
    cfg = wandb.cfg 

    n_cpus = int(sys.argv[1])

    register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002',
                'reward_fun': custom_reward}
    )

    g = gym.make('simglucose-adolescent2-v0')
    env = gym.get_multi_env(n=n_cpus) if n_cpus > 1 else gym.get_single_env()
    model = make_model(env)

    print("Running model")
    model.learn(total_timesteps = 2048 * 500, callback = WandbCallback(name=sys.argv[1], viz=viz, multiprocessing=(n_cpus>1)))
    return model


def main():
    config = {}
    wandb.init(
      project="test-space",
      group="glucose",
      entity="aypan17",
      config=config,
      sync_tensorboard=True
    )
    train()

if __name__ == '__main__':
    main()