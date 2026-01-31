from typing import Any, Dict, Optional

from ...envs import BaseDiscreteEnv
from ..base import RLAgent


class PolicyIter(RLAgent):
    def __init__(self, env: BaseDiscreteEnv, discount_rate: float) -> None:
        super().__init__(env, discount_rate)

    def reset(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(options)

    def forward(self, obs: Any) -> Any: ...

    def policy_eval(self) -> Any: ...

    def policy_impr(self) -> Any: ...
