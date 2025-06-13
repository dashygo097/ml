import warnings
from typing import Optional, overload

from termcolor import colored
from torch import nn
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .bench_backend import benchmark_wikitext
from .evaluator import Evaluator


class LLMEvaluator(Evaluator):
    def __init__(self, model=None, tokenizer=None, device: str = "cpu"):
        super().__init__(device)
        self.load(model, tokenizer, info=False)

    @overload
    def load(
        self, model: Optional[str] = None, tokenizer=None, info: bool = True
    ) -> None:
        self.load(model, tokenizer, info)

    @overload
    def load(self, model: nn.Module, tokenizer=None, info: bool = True) -> None:
        self.load(model, tokenizer, info)

    def bench(self, benchmark: str, log_dict: str = "eval_logs") -> None:
        if benchmark == "wikitext":
            perplexity = benchmark_wikitext(self.model_name, self.device)
            self.logger["perplexity"] = perplexity
            self.save_log(log_dict + "/wikitext")

        """
        elif benchmark == "timer":
            time = benchmark_output_time(self.model, self.tokenizer, self.device)
            self.logger["timer"] = time
            self.save_log(log_dict + "/timer")
        """

    def load(
        self,
        model: Optional[str] | nn.Module = None,
        tokenizer=None,
        info: bool = True,
    ):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model)
            self.model_name = model
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                self.tokenizer = tokenizer

        elif isinstance(model, nn.Module):
            self.model = model
            self.model_name = model.__class__.__name__
            self.tokenizer = tokenizer

        else:
            if info:
                if model is None:
                    warnings.warn(
                        colored(
                            "[WARN] Argument `model` is `None`, NOT LOAD",
                            color="yellow",
                            attrs=["bold"],
                        )
                    )

                if tokenizer is None:
                    warnings.warn(
                        colored(
                            "[WARN] Argument `tokenizer` is `None`, NOT LOAD",
                            color="yellow",
                            attrs=["bold"],
                        )
                    )
