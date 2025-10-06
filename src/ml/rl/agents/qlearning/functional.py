from typing import Any, Callable, Dict, Optional

import torch

from ...envs import BaseDiscreteEnv
from ...trainer import RLTrainArgs, RLTrainer
from ..replay_buffer import ReplayBuffer
from .model import DeepQLearning


class DQNTrainArgs(RLTrainArgs):
    def __init__(self, path_or_dict: str | Dict):
        super().__init__(path_or_dict)
        self.replay_buffer_size: int = self.args.get("replay_buffer_size", 10000)
        self.batch_size: int = self.args.get("batch_size", 32)
        self.target_update_freq: int = self.args.get("target_update_freq", 1000)
        self.learning_starts: int = self.args.get("learning_starts", 1000)
        self.train_freq: int = self.args.get("train_freq", 1)


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
            capacity=args.replay_buffer_size,
            observation_shape=observation_shape,
            device=self.device,
        )

    def step(self) -> Dict[str, Any]:
        action = self.agent(self._obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Store transition in the replay buffer
        self.replay_buffer.append(
            (
                self._obs,
                action,
                reward,
                next_obs,
                terminated,
                truncated,
            )
        )

        loss = torch.tensor(0.0).to(self.device)
        if (
            len(self.replay_buffer) >= self.args.learning_starts
            and self.n_steps % self.args.train_freq == 0
        ):
            # FIXME: impl batch sampling
            batch = self.replay_buffer.sample(self.args.batch_size)
            q_values = (
                self.agent.dqn(batch["obs"])
                .gather(1, batch["action"].unsqueeze(1))
                .squeeze(1)
            )

            with torch.no_grad():
                next_q_values = self.agent.dqn(batch["next_obs"]).max(dim=1)[0]
                next_q_values = next_q_values * (~batch["terminated"])
                target_q_values = (
                    batch["reward"] + self.agent.discount_rate * next_q_values
                )

            loss = self.criterion(q_values, target_q_values)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self._obs = next_obs
        self._info = info
        self._terminated = terminated
        self._truncated = truncated

        if self._terminated:
            self.agent.update_epsilon()

        return {
            "loss": loss.item(),
            "reward": reward,
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
