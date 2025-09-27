from typing import Dict, List, Optional

from openai import OpenAI

from ..client import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, messages: List, model: str, **options) -> Dict[str, str]:
        response = self.client.chat.completions.create(
            model=model, messages=messages, **options
        )
        return response


class OpenAIPlatform:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._client = OpenAIClient(api_key)

    def client(self) -> OpenAIClient:
        return self._client

    def list_availables(self) -> List:
        response = self._client.client.models.list()
        return response.data

    def list_openai_models(self) -> List:
        availables = self.list_availables()
        owned_by_openai = []
        for item in availables:
            owned_by_openai.append(item) if item.owned_by == "openai" else None

        return owned_by_openai

    def list_system_models(self) -> List:
        availables = self.list_availables()
        system_models = []
        for item in availables:
            system_models.append(item) if item.owned_by == "system" else None

        return system_models

    def list_user_models(self) -> List:
        availables = self.list_availables()
        user_models = []
        for item in availables:
            user_models.append(item) if item.owned_by.startswith("user-") else None

        return user_models

    def search_by_id(self, id: str) -> Optional[Dict]:
        availables = self.list_availables()
        for item in availables:
            if item.id == id:
                return item
