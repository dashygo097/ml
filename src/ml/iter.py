class BaseIterator:
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        while True:
            yield self.data
            break
