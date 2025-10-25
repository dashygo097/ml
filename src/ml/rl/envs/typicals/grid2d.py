from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..base import BaseDiscreteEnv


class Grid2DEnv(BaseDiscreteEnv):
    def __init__(self, size: int) -> None:
        super().__init__()

        self.size: int = size
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = gym.spaces.Discrete(5)
        self.action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self._forbidden_area = np.empty((0, 2), dtype=int)
        self._episode_step = 0
        self._max_episode_steps = None

    def get_act_shape(self) -> Tuple[int, ...]:
        return (self.action_space.n,)

    def get_obs_shape(self) -> Tuple[int, ...]:
        return (self.size, self.size)

    def get_obs(self) -> Dict[str, Any]:
        return {"agent": self._agent_location, "target": self._target_location}

    def get_info(self) -> Dict[str, Any]:
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._episode_step = 0
        self._max_episode_steps = None

        observation = self.get_obs()
        info = self.get_info()

        if options is not None:
            if "agent_location" in options.keys():
                self._agent_location = np.clip(
                    options["agent_location"], 0, self.size - 1
                )
            if "target_location" in options.keys():
                self._target_location = np.clip(
                    options["target_location"], 0, self.size - 1
                )
            if "forbidden_area" in options.keys():
                self._set_forbidden_area(options["forbidden_area"])

            if "max_episode_steps" in options.keys():
                self._max_episode_steps = options["max_episode_steps"]

        return observation, info

    def step(self, action) -> Tuple:
        direction = self.action_to_direction[action]

        reward = 0
        if (
            (self._agent_location[0] + direction[0] < 0)
            or (self._agent_location[0] + direction[0] >= self.size)
            or (self._agent_location[1] + direction[1] < 0)
            or (self._agent_location[1] + direction[1] >= self.size)
        ):
            reward -= 1.0

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = terminated or (
            (self._max_episode_steps is not None)
            and (self._episode_step >= self._max_episode_steps)
        )

        if terminated:
            reward += 1.0
        elif np.any(np.all(self._agent_location == self._forbidden_area, axis=1)):
            reward -= 1.0

        observation = self.get_obs()
        info = self.get_info()

        self._episode_step += 1

        return observation, reward, terminated, truncated, info

    def _set_forbidden_area(self, value: np.ndarray | List) -> None:
        self._forbidden_area = np.clip(value, 0, self.size - 1)
