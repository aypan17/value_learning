from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class WandbCallback(BaseCallback):
	"""
	A wandb logging callback that derives from ``BaseCallback``.

	:param verbose: (int) Verbosity level 0: not output 1: info 2: debug
	"""
	def __init__(self, viz=None, multiprocessing=False, verbose=0):
		self.multi = multiprocessing
		self.viz = viz
		super(WandbCallback, self).__init__(verbose)
		# Those variables will be accessible in the callback
		# (they are defined in the base class)
		# The RL model
		# self.model = None  # type: BaseAlgorithm
		# An alias for self.model.get_env(), the environment used for training
		# self.training_env = None	# type: Union[gym.Env, VecEnv, None]
		# Number of time the callback was called
		# self.n_calls = 0	# type: int
		# self.num_timesteps = 0  # type: int
		# local and global variables
		# self.locals = None  # type: Dict[str, Any]
		# self.globals = None  # type: Dict[str, Any]
		# The logger object, used to report things in the terminal
		# self.logger = None  # stable_baselines3.common.logger
		# # Sometimes, for event callback, it is useful
		# # to have access to the parent object
		# self.parent = None  # type: Optional[BaseCallback]

	def _on_training_start(self) -> None:
		"""
		This method is called before the first rollout starts.
		"""
		

	def _on_rollout_start(self) -> None:
		"""
		A rollout is the collection of environment interaction
		using the current policy.
		This event is triggered before collecting new samples.
		"""
		self.episode_true_rewards = []
		self.episode_percent_cost = []
		self.episode_rewards = []
		self.episode_vols = []

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		
		#true_rew = [r for (r, p, v) in out]
		#percent = [p for (r, p, v) in out]
		#vol = [v for (r, p, v) in out]
		#self.episode_true_rewards.append(np.mean(out))
		#self.episode_percent_cost.append(np.mean(percent))
		#self.episode_rewards.append(np.mean(self.training_env.get_reward()))
		#self.episode_vols.append(np.mean(vol))
		#self.viz.record((self.training_env.observation(), self.training_env.last_reward()))
		return True

	def _on_rollout_end(self) -> None:
		"""
		This event is triggered before updating the policy.
		"""
		#acc = self.training_env.get_account_info()[0]
		wandb.log({"calls": self.n_calls})
	#			"true_reward": np.sum(acc['true_reward']).item(),
#				"social responsbility loss (%)": -1 * np.mean(acc['sr_percent']).item(),
	#			"reward": np.sum(acc['reward']).item(),
	#			"volatility cost": np.mean(acc['vol_cost']).item(),
	#			"true volatility cost": np.mean(acc['true_vol_cost']).item()
		#})

	def _on_training_end(self) -> None:
		"""
		This event is triggered before exiting the `learn()` method.
		"""
		pass
