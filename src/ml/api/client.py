from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base

    @abstractmethod
    def chat_completion(
        self, messages: List, model: str, **options
    ) -> Dict[str, Any]: ...
