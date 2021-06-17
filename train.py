from argparse import ArgumentParser
import wandb

def train():
    config = wandb.config
    pass

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-p', '--metric', type=str, default='reward')
    parser.add_argument('-g', '--goal', type=str, default='maximize')
    parser.add_argument('-n', '--name', type=str, default='untitled')
    parser.add_argument('-gpu', '--gpuid', type=int, default=0)

    parser.add_argument('-o', '--optim', type=str, default='adam')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--lr', type=float, default=5e-5)
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-di', '--d_model', type=int, default=128)
    parser.add_argument('-dr', '--dropout', type=float, default=0.2)
   

    args = parser.parse_args()
    return args

def main(args):
    os.environ["WANDB_SILENT"] = "true"

    if args.sweep:
        sweep_config = {
            'method': args.sweep_method,
            'metric': {
              'name': args.metric,
              'goal': args.goal  
            },
            'parameters': {
                'epochs': {
                    'values': [2, 5, 10]
                },
                'batch_size': {
                    'values': [256, 128, 64, 32]
                },
                'dropout': {
                    'values': [0.3, 0.4, 0.5]
                },
                'lr': {
                    'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
                },
                'd_model':{
                    'values':[128,256,512]
                },
                'optimizer': {
                    'values': ['adam', 'sgd']
                },
            }
        }

        sweep_id = wandb.sweep(sweep_config, entity="aypan17", project="value-learning")
        wandb.agent(sweep_id, train)

    else:
        config_defaults = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': args.optim,
            'd_model': args.d_model,
            'dropout': arg.dropout,
        }

        wandb.init(
          project="value-learning",
          entity="aypan17",
          config=config_defaults,
          sync_tensorboard=True
        )
        train()

if __name__ == '__main__':
    args = get_args()
    main(args)