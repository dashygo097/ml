from typing import Any, Dict

import torch

from ...envs import BaseEnv
from ...trainer import RLTrainArgs, RLTrainer
from .model import QLearning


class QLearningTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict):
        super().__init__(path_or_dict)


class QLearningTrainer(RLTrainer):
    def __init__(
        self,
        agent: QLearning,
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
        log = self.agent.update(self._obs, action, lr=self.args.lr)

        self._obs = log["next_obs"]
        self._info = log["info"]
        self._terminated = log["terminated"]
        self._truncated = log["truncated"]

        if self._terminated:
            self.agent.update_epsilon()

        return {
            "reward": log["reward"],
            "loss": log["loss"],
        }

    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "epoch",
            lambda x: {
                "loss": x.get("loss", 0) + result["loss"],
                "reward": x.get("reward", 0) + result["reward"],
            },
            index=self.n_epochs,
        )
