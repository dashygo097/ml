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
        next_obs, reward, self._terminated, self._truncated, info = self.env.step(
            action
        )

        with torch.no_grad():
            future_q = (
                0
                if self._terminated
                else torch.max(self.agent.q_values[tuple(next_obs["agent"])])
            )
            target = reward + self.agent.discount_rate * future_q

            error = target - self.agent.q_values[tuple(self._obs["agent"]) + (action,)]
            loss = error.pow(2)

            self.agent.q_values[tuple(self._obs["agent"]) + (action,)] += (
                self.args.lr * error
            )

        self._obs = next_obs
        self._info = info

        if self._terminated:
            self.agent.update_epsilon()

        return {
            "reward": reward,
            "loss": loss.item(),
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
