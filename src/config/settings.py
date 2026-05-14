from typing import Annotated

from PIL.ImagePath import Path
from pydantic_settings import BaseSettings
from functools import cached_property
from pydantic import computed_field, BeforeValidator


def parse_comma_tuple(v: str | tuple) -> tuple:
    if isinstance(v, str):
        return tuple(v.split(","))
    return v


class Settings(BaseSettings):
    ACTIVATION_FUNCTION: str = "relu"
    DATA_AUGMENTED: bool = False
    MODEL_ROOT_PATH: Path = Path("model")
    SRC_PATH: Path = Path("images")
    IMG_SIZE: Annotated[tuple[int, int], BeforeValidator(parse_comma_tuple)] = (
        1128,
        1226,
    )
    IMG_CONTOUR_TARGET_SIZE: int = 600

    @computed_field
    @cached_property
    def TRAIN_PATH(self) -> Path:
        return self.SRC_PATH / "Train"

    @computed_field
    @cached_property
    def TEST_PATH(self) -> Path:
        return self.SRC_PATH / "Test"

    @computed_field
    @cached_property
    def MODEL_PATH(self) -> Path:
        if self.DATA_AUGMENTED:
            path = self.MODEL_PATH / f"{self.ACTIVATION_FUNCTION}_augmented"
        else:
            path = self.MODEL_PATH / self.ACTIVATION_FUNCTION
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @cached_property
    def MODEL(self) -> Path:
        if self.DATA_AUGMENTED:
            return f"{self.ACTIVATION_FUNCTION}_augmented"
        return self.ACTIVATION_FUNCTION

    model_config = {"env_file": ".env.local"}


config = Settings()
