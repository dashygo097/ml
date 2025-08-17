from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from gymnasium import Env


class BaseEnv(ABC, Env):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_obs(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_info(self) -> Dict[str, Any]: ...

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

    @abstractmethod
    def step(self, action) -> Tuple: ...


class BaseDiscreteEnv(BaseEnv):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_act_shape(self) -> Tuple[int, ...]: ...

    @abstractmethod
    def get_obs_shape(self) -> Tuple[int, ...]: ...
