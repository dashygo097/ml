from trtrt import FrontEnd, Scene

from .base import BaseEnv


class BaseScene(Scene):
    def __init__(self, env: BaseEnv, maximum: int = 128) -> None:
        self.env = env
        super().__init__(maximum)


class BaseFrontEnd(FrontEnd):
    def __init__(self, name: str, res: tuple[int, int]) -> None:
        super().__init__(name, res)
