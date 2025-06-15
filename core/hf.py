from torch.utils.data import Dataset


class HFDatasetWrapper(Dataset):
    def __init__(self, dataset, tokenizer, feature: str, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.feature = feature
        self.max_length = max_length

        other_columns = [col for col in dataset.column_names]
        self.dataset = dataset.map(self._tokenize_fn, batched=True)
        self.dataset = self.dataset.remove_columns(other_columns)
        self.dataset.set_format("torch")

    def _tokenize_fn(self, example):
        return self.tokenizer(
            example[self.feature],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if "labels" in self.dataset.column_names:
            return (
                self.dataset[idx]["input_ids"],
                self.dataset[idx]["attention_mask"],
                self.dataset[idx]["labels"],
            )
        else:
            return (
                self.dataset[idx]["input_ids"],
                self.dataset[idx]["attention_mask"],
            )
