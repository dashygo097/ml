from typing import Any, Dict, Optional

import torch

from ...envs import BaseDiscreteEnv
from ...trainer import RLTrainArgs, RLTrainer
from .model import QLearning


class QLearningTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict):
        super().__init__(path_or_dict)


class QLearningTrainer(RLTrainer):
    def __init__(
        self,
        agent: QLearning,
        env: BaseDiscreteEnv,
        args: RLTrainArgs,
        optimizer: Optional[type] = None,
        scheduler: Optional[type] = None,
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
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        q_idx = tuple(self._obs["agent"]) + (action,)
        future_idx = tuple(next_obs["agent"])

        with torch.no_grad():
            future_q = 0 if terminated else self.agent.q_values[future_idx].max()
            target = reward + self.agent.discount_rate * future_q
            error = target - self.agent.q_values[q_idx]
            loss = error * error

            self.agent.q_values[q_idx] += self.args.lr * error

        self._obs = next_obs
        self._info = info
        self._terminated = terminated
        self._truncated = truncated

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
