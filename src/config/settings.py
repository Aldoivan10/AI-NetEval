from pathlib import Path
from pydantic_settings import BaseSettings
from functools import cached_property
from pydantic import computed_field


def parse_comma_tuple(v: str | tuple) -> tuple:
    if isinstance(v, str):
        return tuple(v.split(","))
    return v


class Settings(BaseSettings):
    ACTIVATION_FUNCTION: str = "relu"
    DATA_AUGMENTED: bool = False
    MODEL_ROOT_PATH: Path = Path("model")
    MODEL_NAME: str = "best_model.keras"
    SRC_PATH: Path = Path("images")
    # Ancho x Alto para normalizar los exámenes escaneados
    IMG_SIZE: tuple[int, int] = (1128, 1226)
    # Tamaño mínimo de los recuadros de las respuestas
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
            path = self.MODEL_ROOT_PATH / f"{self.ACTIVATION_FUNCTION}_augmented"
        else:
            path = self.MODEL_ROOT_PATH / self.ACTIVATION_FUNCTION
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @computed_field
    @cached_property
    def MODEL(self) -> Path:
        return self.MODEL_PATH / self.MODEL_NAME

    model_config = {"env_file": ".env.local"}


config = Settings()
