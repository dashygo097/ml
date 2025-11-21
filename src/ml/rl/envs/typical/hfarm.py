from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..base import BaseDiscreteEnv

# Global Info
HFARM2D_SIZE: Tuple[int, int] = (12, 22)
HFRAM2D_WATER_AREAS: List[Tuple[int, int]] = [
    (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
    (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), 
    (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
    (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), 
    (8, 4), (8, 5), (8, 6), 
    (18, 1), (18, 2),
    (19, 1), (19, 2), (19, 3),
    (20, 2), (20, 3)
]

# Action Space
HFARM2D_MOVE_ACTIONS : List[str] = [
    "IDLE",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "UP_LEFT",
    "UP_RIGHT",
    "DOWN_LEFT",
    "DOWN_RIGHT",
]
HFARM2D_PLANT_ACTIONS: List[str] = [
    "PLANT_STRAWBERRY",
    "PLANT_GRAPE",
    "PLANT_WHEAT",
    "PLANT_LOTUS",
    "PLANT_PUMPKIN"
]
HFARM2D_FARM_ACTIONS: List[str] = [
    "HARVEST",
    "FILLWATER",
    "WATER"
]

class HFarm2DEnv(BaseDiscreteEnv):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                # Agent position in the grid
                "agent_pos": gym.spaces.Box(low=0, high=np.array(HFARM2D_SIZE) - 1, shape=(2,), dtype=np.int32),
                # States of planted crops in the grid (not planted, * is growing, * can be harvested)
                "planted_crops": gym.spaces.MultiBinary((HFARM2D_SIZE[0], HFARM2D_SIZE[1], len(HFARM2D_PLANT_ACTIONS) * 2 + 1)),
            }
        )
        self.action_space = gym.spaces.Discrete(len(HFARM2D_MOVE_ACTIONS) + len(HFARM2D_PLANT_ACTIONS) + len(HFARM2D_FARM_ACTIONS))
        self.action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]),
            5: np.array([-1, -1]),
            6: np.array([-1, 1]),
            7: np.array([1, -1]),
            8: np.array([1, 1]),
        }

        self._agent_location = np.array([0, 0], dtype=np.int32)
        
        self._episode_step = 0
        self._max_episode_steps = 1 

    def get_act_shape(self) -> Tuple[int, ...]:
        return (self.action_space.n,)

    def get_obs_shape(self) -> Tuple[int, ...]:
        return HFARM2D_SIZE

    def get_obs(self) -> Dict[str, Any]:
        return {"agent_pos": self._agent_location.copy}

    def get_info(self) -> Dict[str, Any]:
        return {
            "shortest_path_to_water": self._compute_shortest_path_to_water(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, HFARM2D_SIZE, size=(2,), dtype=np.int32)

        self._episode_step = 0

        observation = self.get_obs()
        info = self.get_info()

        if options is not None:
            if "max_episode_steps" in options.keys():
                self._max_episode_steps = options["max_episode_steps"]

        return observation, info

    def step(self, action: Any) -> Tuple:
        move_action_space_size = len(HFARM2D_MOVE_ACTIONS)
        plant_action_space_size = len(HFARM2D_PLANT_ACTIONS)
        farm_action_space_size = len(HFARM2D_FARM_ACTIONS)

        reward = 0

        if action < move_action_space_size:
            direction = self.action_to_direction[action]
            new_location = self._agent_location + direction
            new_location = np.clip(new_location, [0, 0], np.array(HFARM2D_SIZE) - 1)
            self._agent_location = new_location
        elif action < move_action_space_size + plant_action_space_size:

            

        terminated = False 
        truncated = self._episode_step >= self._max_episode_steps

        observation = self.get_obs()
        info = self.get_info()

        self._episode_step += 1

        return observation, reward, terminated, truncated, info


    def _compute_shortest_path_to_water(self) -> int:
        min_distance = float('inf')
        for water_pos in HFRAM2D_WATER_AREAS:
            distance = np.sum(np.abs(self._agent_location - np.array(water_pos)))
            if distance < min_distance:
                min_distance = distance
        return int(min_distance)
