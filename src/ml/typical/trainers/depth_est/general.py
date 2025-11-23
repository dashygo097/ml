from typing import Any, Callable, Dict

import torch
from termcolor import colored
from torch import nn
from torch.utils.data import Dataset

from ....trainer import TrainArgs, Trainer


class DepthEstTrainArgs(TrainArgs):
    def __init__(self, path_or_dict: str | Dict[str, Any]) -> None:
        super().__init__(path_or_dict)


class DepthEstTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        loss_fn: Callable,
        args: DepthEstTrainArgs,
        optimizer=None,
        scheduler=None,
        valid_ds=None,
    ) -> None:
        super().__init__(model, dataset, loss_fn, args, optimizer, scheduler, valid_ds)

    def step(self, batch) -> Dict[str, Any]:
        (imgs, labels), _ = batch
        imgs, labels = (
            imgs.to(self.device),
            labels.to(self.device),
        )
        self.optimizer.zero_grad()
        logits = self.model(imgs)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


    def step_info(self, result: Dict[str, Any]) -> None:
        self.logger.op(
            "step",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_steps
        )

        if self.n_steps % 10 == 0 and self.n_steps > 0:
            step_loss = self.logger.content.step[f'{self.n_steps}']['loss']
            
            print(
                f"(Step {self.n_steps}) "
                + colored("loss", "yellow")
                + f": {step_loss:.4f}"
            )

        # epoch
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) + result["loss"]},
            index=self.n_epochs,
        )

    def epoch_info(self) -> None:
        self.logger.op(
            "epoch",
            lambda x: {"loss": x.get("loss", 0) / len(self.data_loader)},
            index=self.n_epochs,
        )
        print(
            f"(Epoch {self.n_epochs}) "
            + colored("loss", "yellow")
            + f": {self.logger.content.epoch[f'{self.n_epochs}']['loss']:.4f}"
        )

    def validate(self) -> None:
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        total_abs_rel = 0.0
        total_sq_rel = 0.0
        total_rmse = 0.0
        total_rmse_log = 0.0
        total_delta1 = 0.0
        total_delta2 = 0.0
        total_delta3 = 0.0
        
        for batch in self.valid_data_loader:
            (imgs, labels), info = batch
            imgs, labels = (
                imgs.to(self.device),
                labels.to(self.device),
            )

            
            with torch.no_grad():
                preds = self.model(imgs)  # [B, 1, H, W]
                loss = self.loss_fn(preds, labels)
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                valid_mask = labels > 0
                
                if valid_mask.sum() > 0:
                    pred_valid = preds[valid_mask]
                    label_valid = labels[valid_mask]
                    
                    abs_rel = torch.mean(torch.abs(pred_valid - label_valid) / label_valid)
                    total_abs_rel += abs_rel.item() * batch_size
                    
                    sq_rel = torch.mean(((pred_valid - label_valid) ** 2) / label_valid)
                    total_sq_rel += sq_rel.item() * batch_size
                    
                    rmse = torch.sqrt(torch.mean((pred_valid - label_valid) ** 2))
                    total_rmse += rmse.item() * batch_size
                    
                    rmse_log = torch.sqrt(torch.mean((torch.log(pred_valid + 1e-8) - torch.log(label_valid + 1e-8)) ** 2))
                    total_rmse_log += rmse_log.item() * batch_size
                    
                    thresh = torch.max(pred_valid / label_valid, label_valid / pred_valid)
                    delta1 = (thresh < 1.25).float().mean()
                    delta2 = (thresh < 1.25 ** 2).float().mean()
                    delta3 = (thresh < 1.25 ** 3).float().mean()
                    
                    total_delta1 += delta1.item() * batch_size
                    total_delta2 += delta2.item() * batch_size
                    total_delta3 += delta3.item() * batch_size
        
        val_loss = total_loss / total_samples
        abs_rel = total_abs_rel / total_samples
        sq_rel = total_sq_rel / total_samples
        rmse = total_rmse / total_samples
        rmse_log = total_rmse_log / total_samples
        delta1 = total_delta1 / total_samples
        delta2 = total_delta2 / total_samples
        delta3 = total_delta3 / total_samples
        
        self.logger.log(
            "valid",
            {
                "val_loss": val_loss,
                "abs_rel": abs_rel,
                "sq_rel": sq_rel,
                "rmse": rmse,
                "rmse_log": rmse_log,
                "delta1": delta1,
                "delta2": delta2,
                "delta3": delta3,
            },
            self.n_epochs,
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Validation Epoch {self.n_epochs}")
        print(f"{'='*60}")
        print(f"{colored('Loss', 'red')}: {val_loss:.4f}")
        print(f"\n{colored('Error Metrics:', 'yellow')}")
        print(f"  Abs Rel Error: {abs_rel:.4f}")
        print(f"  Sq Rel Error:  {sq_rel:.4f}")
        print(f"  RMSE:          {rmse:.4f}")
        print(f"  RMSE log:      {rmse_log:.4f}")
        print(f"\n{colored('Accuracy Metrics (higher is better):', 'green')}")
        print(f"  δ < 1.25:      {delta1:.4f} ({delta1*100:.2f}%)")
        print(f"  δ < 1.25²:     {delta2:.4f} ({delta2*100:.2f}%)")
        print(f"  δ < 1.25³:     {delta3:.4f} ({delta3*100:.2f}%)")
        print(f"{'='*60}\n")
        
