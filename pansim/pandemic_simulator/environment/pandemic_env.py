# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import numpy as np
from copy import deepcopy

import gym
from gym import spaces

from .done import DoneFunction
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, \
    InfectionSummary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ['PandemicGymEnv', 'PandemicPolicyGymEnv']


class PandemicGymEnv(gym.Env):
    """A gym environment interface wrapper for the Pandemic Simulator."""

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _obs_history_size: int
    _sim_steps_per_regulation: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: PandemicObservation
    _last_reward: float

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 true_reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 24,
                 non_essential_business_location_ids: Optional[List[LocationID]] = None,
                 ):
        """
        :param pandemic_sim: Pandemic simulator instance
        :param pandemic_regulations: A sequence of pandemic regulations
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param sim_steps_per_regulation: number of sim_steps to run for each regulation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._obs_history_size = obs_history_size
        self._sim_steps_per_regulation = sim_steps_per_regulation

        if non_essential_business_location_ids is not None:
            for loc_id in non_essential_business_location_ids:
                assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
                                  NonEssentialBusinessLocationState)
        self._non_essential_business_loc_ids = non_essential_business_location_ids

        self._reward_fn = reward_fn
        self._true_reward_fn = true_reward_fn 

        self._done_fn = done_fn

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(pandemic_sim.poll()),)
        )
        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    obs_history_size: int = 1,
                    non_essential_business_location_ids: Optional[List[LocationID]] = None,
                    ) -> 'PandemicGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        sim = PandemicSim.from_config(sim_config, sim_opts)

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, 1, .1, 0.02]
        )

        return PandemicGymEnv(pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              done_fn=done_fn,
                              obs_history_size=obs_history_size,
                              non_essential_business_location_ids=non_essential_business_location_ids)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> PandemicObservation:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    @property
    def get_true_reward(self) -> float:
        return self._last_true_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
                                              self._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self._last_observation
        self._last_reward = self._reward_fn.calculate_reward(prev_obs, action, obs) if self._reward_fn else 0.
        self._last_true_reward = self._true_reward_fn.calculate_reward(prev_obs, action, obs) 
        done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
        self._last_observation = obs

        #return self._last_observation, self._last_reward, done, {}
        return self._pandemic_sim.poll(), self._last_reward, done, {}

    def reset(self) -> PandemicObservation:
        self._pandemic_sim.reset()
        self._last_observation = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)
        self._last_reward = 0.0
        self._last_true_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        #return self._last_observation
        return self._pandemic_sim.poll()

    def render(self, mode: str = 'human') -> bool:
        pass

class PandemicPolicyGymEnv(PandemicGymEnv):

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 true_reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 24,
                 non_essential_business_location_ids: Optional[List[LocationID]] = None,
                 ):

        super().__init__(pandemic_sim,
                 pandemic_regulations,
                 reward_fn,
                 true_reward_fn,
                 done_fn,
                 obs_history_size,
                 sim_steps_per_regulation,
                 non_essential_business_location_ids,
                )
        

    @classmethod
    def from_config(cls: Type['PandemicPolicyGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    obs_history_size: int = 1,
                    non_essential_business_location_ids: Optional[List[LocationID]] = None,
                    alpha: float = 0.4,
                    beta: float = 1,
                    gamma: float = 0.1,
                    delta: float = 0.02,
                    ) -> 'PandemicPolicyGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param raw_regulations: The raw regulations output by regulation_network before processing
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        sim = PandemicSim.from_config(sim_config, sim_opts)

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[alpha, beta, gamma, delta]
        )

        true_reward_fn = SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, 1, .1, 0.02]
        )

        return PandemicPolicyGymEnv(pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              true_reward_fn=true_reward_fn,
                              done_fn=done_fn,
                              obs_history_size=obs_history_size,
                              non_essential_business_location_ids=non_essential_business_location_ids)

    def get_single_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e

    def get_multi_env(self, n=10):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e
