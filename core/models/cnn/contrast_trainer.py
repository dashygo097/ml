from .trainer import BasicCNNTrainArgs, BasicCNNTrainer


class ContrastTrainerArgs(BasicCNNTrainArgs):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.contrast_margin = self.args["contrast"].get("margin", 0.5)
        self.contrast_weight = self.args["contrast"].get("weight", 1.0)
