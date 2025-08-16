from typing import Any, Dict, Optional

from ...envs import BaseEnv
from ..base import RLAgent


class ValueIter(RLAgent):
    def __init__(self, env: BaseEnv, discount_rate: float = 0.99) -> None:
        super().__init__(env, discount_rate)

    def reset(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(options)

    def forward(self, obs: Any) -> Any: ...
