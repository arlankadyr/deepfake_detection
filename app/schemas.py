"""API response schemas."""

from pydantic import BaseModel, Field


class AvailableRun(BaseModel):
    train_size: int
    seed: int


class ModelInfo(BaseModel):
    model: str
    display_name: str
    architecture: str
    parameter_count: int = Field(ge=1)
    training_data: str
    available_runs: list[AvailableRun]
    training_sizes: list[int]
    seeds_by_size: dict[str, list[int]]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    training_sizes: list[int]


class Prediction(BaseModel):
    model: str
    fake_probability: float = Field(ge=0, le=1)
    label: str
    inference_ms: float = Field(ge=0)


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    models_agree: bool
