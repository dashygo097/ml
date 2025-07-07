class BaseIterator:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1

    def __iter__(self):
        while True:
            yield self.data
            break
