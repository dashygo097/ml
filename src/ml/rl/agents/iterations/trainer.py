from typing import Any, Dict, Optional

import torch

from ...envs import BaseEnv
from ...trainer import RLTrainArgs, RLTrainer
from .policy_iter import PolicyIter
from .value_iter import ValueIter


class PolicyIterTrainer(RLTrainer):
    def __init__(
        self,
        agent: PolicyIter,
        env: BaseEnv,
        args: RLTrainArgs,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            agent=agent,
            env=env,
            args=args,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def step(self) -> Dict[str, Any]:
        action = self.agent(self._obs)
        next_obs, reward, self._terminated, self._truncated, info = self.env.step(
            action
        )

        with torch.no_grad():
            ...

        return {}



class ValueIterTrainer(RLTrainer):
    def __init__(
        self,
        agent: ValueIter,
        env: BaseEnv,
        args: RLTrainArgs,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__(
            agent=agent,
            env=env,
            args=args,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def step(self) -> Dict[str, Any]:
        action = self.agent(self._obs)
        next_obs, reward, self._terminated, self._truncated, info = self.env.step(
            action
        )
        with torch.no_grad():
            ...
        return {}
