import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from datetime import datetime
import time
import torch
import sys

import argparse

import wandb
from finrl.callback import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.env.env_stocktrading_v2 import StockTradingEnvV2
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_plot, backtest_stats

import multiprocessing

from pprint import pprint
import os
if not os.path.exists("./fin_results/" + config.DATA_SAVE_DIR):
	os.makedirs("./fin_results/" + config.DATA_SAVE_DIR)
if not os.path.exists("./fin_results/" + config.TRAINED_MODEL_DIR):
	os.makedirs("./fin_results/" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./fin_results/" + config.TENSORBOARD_LOG_DIR):
	os.makedirs("./fin_results/" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./fin_results/" + config.RESULTS_DIR):
	os.makedirs("./fin_results/" + config.RESULTS_DIR)


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a FinRL simulation.",
        epilog="python train.py")

# Environment args
    parser.add_argument(
        '--start_date', type=str, default="2009-01-01",
        help='the start date of training in YYYY-MM-DD format')
    parser.add_argument(
        '--mid_date', type=str, default="2019-01-01",
        help='the end of training and the start of trading in YYYY-MM-DD format')
    parser.add_argument(
        '--end_date', type=str, default="2021-01-01",
        help='the end date of trading in YYYY-MM-DD format')
    parser.add_argument(
        '--ticker', type=str, default="dow",
        help="which set of stocks to trade on. currently supports 'dow' or '30', 'nasdaq' or '100', 'sp' or '500'")
    parser.add_argument(
        '--principal', type=float, default=1e6,
        help='the principal to trade with')
    parser.add_argument(
        '--cash_threshold', type=float, default=0,
        help='the minimum amount of cash that must be held, lest a penalty is incurred')
    parser.add_argument(
        '--low_cash_penalty', type=float, default=0,
        help="the reward penalty for having cash < CASH_THRESHOLD; penalty of 1 gives r' = r-1")
    parser.add_argument(
        '--cash_penalty_proportion', type=float, default=0.2,
        help='the penalty for holding onto cash')
    parser.add_argument(
        '--vol_multiplier', type=float, default=0.0,
        help='how much the model is rewarded for trading in a volatile market (typically negative)')
    parser.add_argument(
        '--true_vol_multiplier', type=float, default=0.0,
        help='how much the model is rewarded for trading in a volatile market (typically negative); calculated in true_reward')

# Policy args
    parser.add_argument(
        '--num_steps', type=int, default=1024*100,
        help='how many total steps to perform learning over')
    parser.add_argument(
        '--eval_freq', type=int, default=500,
        help='how frequently to evaluate')
    parser.add_argument(
        '--rollout_size', type=int, default=1024,
        help='how many steps are in a training batch.')
    parser.add_argument(
        '--ent', type=float, default=0.0,
        help='the entropy coefficient in PPO')
    parser.add_argument(
        '--bs', type=int, default=1024,
        help='batch size')
    parser.add_argument(
        '--lr', type=float, default=0.000005,
        help='the learning rate')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='the discount factor')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='hidden dimension of each FC layer')
    parser.add_argument(
        '--n_layers', type=int, default=5,
        help='number of hidden layers in model')
    parser.add_argument(
        '--save_path', type=str, default=datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
        help='model will be stored as ./trained_models/{save_path}')

    return parser.parse_known_args(args)[0]

def preprocess():
	cfg = wandb.config
	print(wandb.config)
	if cfg.ticker == 'dow' or cfg.ticker == '30':
		ticker = config.DOW_30_TICKER
	elif cfg.ticker == 'nasdaq' or cfg.ticker == '100':
		ticker = config.NAS_100_TICKER
	elif cfg.ticker == 'sp' or cfg.ticker == '500':
		ticker = config.SP_500_TICKER
	else:
		raise NotImplementedError()
	
	df = YahooDownloader(start_date = cfg.start_date,
					 end_date = cfg.end_date,
					 ticker_list = ticker).fetch_data()
	fe = FeatureEngineer(
					use_technical_indicator=True,
					tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
					use_turbulence=True,
					user_defined_feature = False)
	processed = fe.preprocess_data(df)
	processed['log_volume'] = np.log(processed.volume*processed.close)
	processed['change'] = (processed.close-processed.open)/processed.close
	processed['daily_variance'] = (processed.high-processed.low)/processed.close
	return processed

def make_env(data, multi=False):
	cfg = wandb.config
	train = data_split(data, cfg.start_date, cfg.mid_date)
	trade = data_split(data, cfg.mid_date, cfg.end_date)

	information_cols = ['daily_variance', 'change', 'log_volume', 'close','day', 
						'macd', 'rsi_30', 'cci_30', 'dx_30', 'turbulence']

	train_gym = StockTradingEnvCashpenalty(df = train,initial_amount = cfg.principal,hmax = 5000, 
									cash_penalty_proportion=cfg.cash_penalty_proportion, 
									cache_indicator_data=True,
									vol_multiplier=cfg.vol_multiplier,
									true_vol_multiplier=cfg.true_vol_multiplier,
									moral = int(sys.argv[1]), 
									env = int(sys.argv[2]),
									social = int(sys.argv[3]),
									daily_information_cols = information_cols, 
									print_verbosity = 5000, random_start = True)


	trade_gym = StockTradingEnvCashpenalty(df = trade,initial_amount = cfg.principal,hmax = 5000, 
									cash_penalty_proportion=cfg.cash_penalty_proportion,
									cache_indicator_data=True,
									vol_multiplier=cfg.vol_multiplier,
									true_vol_multiplier=cfg.true_vol_multiplier,
									moral = int(sys.argv[1]), 
									env = int(sys.argv[2]),
									social = int(sys.argv[3]),
									daily_information_cols = information_cols, 
									print_verbosity = 5000, random_start = False)

	# for this example, let's do multiprocessing with n_cores-2
	if multi:
		n_cores = int(sys.argv[4])
		print(f"using {n_cores} cores")
		#this is our training env. It allows multiprocessing
		env_train, _ = train_gym.get_multiproc_env(n = n_cores)

	else:
		env_train, _ = train_gym.get_sb_env()

	#this is our observation environment. It allows full diagnostics
	env_trade, _ = trade_gym.get_sb_env()	 

	return train_gym, env_train, trade_gym, env_trade

def make_model(env_train):
	cfg = wandb.config
	agent = DRLAgent(env = env_train)

	# from torch.nn import Softsign, ReLU
	ppo_params ={'n_steps': cfg.rollout_size, 
				 'ent_coef': cfg.ent, 
				 'learning_rate': cfg.lr, 
				 'batch_size': cfg.bs, 
				'gamma': cfg.gamma}

	policy_kwargs = {
	#	  "activation_fn": ReLU,
		"net_arch": [cfg.d_model] * cfg.n_layers#[1024, 1024,1024, 1024,  1024], 
	#	  "squash_output": True
	}

	model = agent.get_model("ppo",	
							model_kwargs = ppo_params, 
							policy_kwargs = policy_kwargs, verbose = 0)

	return model

def train(model, env_trade):
	cfg = wandb.config
	# model = model.load("scaling_reward.model", env = env_train)
	model.learn(total_timesteps = cfg.num_steps, 
			eval_env = env_trade, 
			eval_freq = cfg.eval_freq,
			log_interval = cfg.eval_freq, 
			tb_log_name = 'test',
			n_eval_episodes = 1,
			callback = WandbCallback())
	#model.save("different.model")
	return model


def evaluate(model, test_gym):
	cfg = wandb.config
	test_gym.hmax = 500
	df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,environment = test_gym)
	perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
	print("*" * 10 + " Backtesting " + "*" * 10)
	print(perf_stats_all)

	print("*" * 10 + " Compare to DOW " + "*" * 10)
	backtest_plot(df_account_value, 
			 baseline_ticker = '^DJI', 
			 baseline_start = cfg.mid_date,
			 baseline_end = cfg.end_date, value_col_name = 'total_assets')

def main(args):
	
	conf = parse_args(args)
	
	wandb.init(project="value-learning", entity="aypan17", group="fin", config=conf, sync_tensorboard=True)

	t1 = time.time()
	data = preprocess()
	train_gym, env_train, trade_gym, env_trade = make_env(data, multi=True)
	print(type(env_train))
	model = make_model(env_train)
	print(type(model.env))
	t2 = time.time()
	print(f"PREPROCESS TIME: {t2-t1}")
	train(model, env_trade)
	t3 = time.time()
	print(f"TRAIN TIME: {t3-t2}")
	model = evaluate(model, trade_gym)
	t4 = time.time()
	print(f"EVAL TIME: {t4-t3}")

#def test():
#	print(f'CUDA:{torch.cuda.is_available()}')
	
	
if __name__ == '__main__':
	print(f'CUDA:{torch.cuda.is_available()}')
	main(sys.argv[1:])




