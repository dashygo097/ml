from typing import Any, Dict

import torch

from ...envs import BaseEnv
from ...trainer import RLTrainArgs, RLTrainer
from .policy_iter import PolicyIter
from .value_iter import ValueIter


class PolicyIterTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)


class ValueIterTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict) -> None:
        super().__init__(path_or_dict)


class PolicyIterTrainer(RLTrainer):
    def __init__(
        self,
        agent: PolicyIter,
        env: BaseEnv,
        args: RLTrainArgs,
        optimizer=None,
        scheduler=None,
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

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {},
            index=self.n_epochs,
        )


class ValueIterTrainer(RLTrainer):
    def __init__(
        self,
        agent: ValueIter,
        env: BaseEnv,
        args: RLTrainArgs,
        optimizer=None,
        scheduler=None,
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

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {},
            index=self.n_epochs,
        )
