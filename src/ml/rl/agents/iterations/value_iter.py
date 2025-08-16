from ...envs import BaseEnv
from ..base import RLAgent


class ValueIterationAgent(RLAgent):
    def __init__(self, env: BaseEnv, discount_rate: float = 0.99) -> None:
        super().__init__(env, discount_rate)
