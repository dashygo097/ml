from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
