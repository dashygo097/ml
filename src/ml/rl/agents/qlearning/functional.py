from typing import Any, Callable, Dict, Optional

import torch

from ...envs import BaseDiscreteEnv
from ...trainer import RLTrainArgs, RLTrainer
from .model import DeepQLearning
from .replay_buffer import ReplayBuffer


class DQNTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict):
        super().__init__(path_or_dict)
        self.replay_buffer_size = self.args.get("replay_buffer_size", 10000)
        self.batch_size = self.args.get("batch_size", 32)
        self.target_update_freq = self.args.get("target_update_freq", 1000)
        self.learning_starts = self.args.get("learning_starts", 1000)
        self.train_freq = self.args.get("train_freq", 1)


class DQNTrainer(RLTrainer):
    def __init__(
        self,
        agent: DeepQLearning,
        env: BaseDiscreteEnv,
        criterion: Callable,
        args: DQNTrainArgs,
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

        self.criterion = criterion

        # Initialize replay buffer
        observation_shape = env.get_obs_shape()
        self.replay_buffer = ReplayBuffer(
            capacity=args.replay_buffer_size, obs_shape=observation_shape
        )

    def step(self) -> Dict[str, Any]:
        action = self.agent(self._obs)
        log = self.agent.update(
            self._obs, action, loss_fn=self.criterion, optimizer=self.optimizer
        )

        ...

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
