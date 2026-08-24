from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    holdout_dir: str
    raw_file: str
    processed_file: str
    holdout_file: str


class GenerationConfig(BaseModel):
    n_samples: int = Field(default=100000)
    n_features: int = Field(default=40)
    n_informative: int = Field(default=20)
    n_redundant: int = Field(default=10)
    n_categorical: int = Field(default=10)
    class_sep: float = Field(default=0.8)
    flip_y: float = Field(default=0.05)
    random_state: int = Field(default=42)
    new_world_random_state: int = Field(default=88)
    holdout_samples: int = Field(default=20000)
    imbalance_ratio: float = Field(default=0.05)


class ModelConfig(BaseModel):
    models_dir: str


class TrainingConfig(BaseModel):
    random_state: int = Field(default=42)


class ProjectConfig(BaseModel):
    data: DataConfig
    generation: GenerationConfig
    model: ModelConfig
    training: TrainingConfig
