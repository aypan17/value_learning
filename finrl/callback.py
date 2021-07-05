from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class WandbCallback(BaseCallback):
	"""
	A wandb logging callback that derives from ``BaseCallback``.

	:param verbose: (int) Verbosity level 0: not output 1: info 2: debug
	"""
	def __init__(self, multiprocessing=False, verbose=0):
		self.multi = multiprocessing
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

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		out = self.training_env.get_true_reward()
		true_rew = [r for (r, p) in out]
		percent = [p for (r, p) in out]
		self.episode_true_rewards.append(np.mean(true_rew))
		self.episode_percent_cost.append(np.mean(percent))
		self.episode_rewards.append(np.mean(self.training_env.get_reward()))
		
		return True

	def _on_rollout_end(self) -> None:
		"""
		This event is triggered before updating the policy.
		"""
		wandb.log({
				"true_reward": np.sum(self.episode_true_rewards).item(),
				"percent_cost": np.mean(self.episode_percent_cost).item(),
				"reward": np.sum(self.episode_rewards).item()
		})

	def _on_training_end(self) -> None:
		"""
		This event is triggered before exiting the `learn()` method.
		"""
		pass
