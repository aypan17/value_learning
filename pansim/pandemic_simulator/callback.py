from stable_baselines3.common.callbacks import BaseCallback
from pandemic_simulator.environment.interfaces import sorted_infection_summary

import wandb
import numpy as np

class WandbCallback(BaseCallback):
    """
    A wandb logging callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, n_ppl=100, viz=None, true_viz=None, eval_freq=10, multiprocessing=False, verbose=0):
        
        self.n_ppl = n_ppl
        self.viz = viz
        self.true_viz = true_viz
        self.eval_freq = eval_freq
        self.multi = multiprocessing

        self.record = False
        super(WandbCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
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
        print(self.n_calls)
        self.episode_rewards = []
        self.episode_true_rewards = []
        self.episode_infection_data = np.array([[0, 0, 0, 0, 0]])
        self.episode_threshold = []

        self.record = ((self.n_calls+1) % self.eval_freq == 0)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        list_obs = self.training_env.get_attr("observation") if self.multi else [self.training_env.get_attr("observation")]
        rew = np.mean(self.training_env.get_attr("last_reward"))
        true_rew = np.mean(self.training_env.get_attr("get_true_reward"))
        infection_data = np.zeros((1, 5))
        threshold_data = np.zeros(len(list_obs))
        for obs in list_obs:
            infection_data += np.squeeze(obs.global_infection_summary, axis=0) 
            threshold_data += np.squeeze(obs.infection_above_threshold)

        self.episode_rewards.append(rew)
        self.episode_true_rewards.append(true_rew)
        self.episode_infection_data = np.concatenate([self.episode_infection_data, infection_data / len(list_obs)])
        self.episode_threshold.append(np.sum(threshold_data) / len(list_obs))
        
        if self.record:
            self.viz.record((obs, rew))
            self.true_viz.record((obs, true_rew))
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print(self.episode_infection_data)
        assert False
        infection_summary = np.mean(self.episode_infection_data, axis=0)
        wandb.log({"reward": np.mean(self.episode_rewards), 
                   "true_reward": np.mean(self.episode_true_rewards),
                   "proportion_critical": infection_summary[0] / self.n_ppl,
                   "proportion_dead": infection_summary[1] / self.n_ppl,
                   "proportion_infected": infection_summary[2] / self.n_ppl,
                   "proportion_healthy": infection_summary[3] / self.n_ppl,
                   "proportion_recovered": infection_summary[4] / self.n_ppl,
                   "time_over_threshold": np.mean(self.episode_threshold),
                   })
        if self.record:
            self.viz.plot()
            self.true_viz.plot(is_true=True)
            self.viz.reset()
            self.true_viz.reset()
    


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
