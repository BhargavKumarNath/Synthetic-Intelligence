import pandas as pd
import pandera as pa
import pytest

from synthetic_intelligence.config import GenerationConfig
from synthetic_intelligence.data.generator import SyntheticDatasetGenerator
from synthetic_intelligence.data.validator import validate_dataset


@pytest.fixture
def config():
    # Use smaller numbers for fast testing
    return GenerationConfig(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_categorical=3,
        imbalance_ratio=0.1,
        holdout_samples=500,
    )


@pytest.fixture
def generator(config):
    return SyntheticDatasetGenerator(config)


def test_generate_base_data(generator, config):
    df = generator.generate_base_data()

    assert len(df) == config.n_samples
    assert "target" in df.columns

    # It should be approximately balanced, but flip_y causes variance
    minority_count = len(df[df["target"] == 1])
    majority_count = len(df[df["target"] == 0])
    assert abs(minority_count - majority_count) < (config.n_samples * 0.1)

    # Check categorical columns exist
    for i in range(config.n_categorical):
        assert f"cat_{i}" in df.columns
        assert df[f"cat_{i}"].dtype.name == "category"

    # Check numerical columns exist
    num_features = config.n_features - config.n_categorical
    for i in range(num_features):
        assert f"num_{i}" in df.columns
        assert df[f"num_{i}"].dtype == float


def test_create_imbalanced_dataset(generator):
    df_base = generator.generate_base_data()
    df_imbalanced = generator.create_imbalanced_dataset(df_base)

    majority_count = len(df_imbalanced[df_imbalanced["target"] == 0])
    minority_count = len(df_imbalanced[df_imbalanced["target"] == 1])

    total = majority_count + minority_count
    ratio = minority_count / total

    # Imbalance ratio should be roughly config.imbalance_ratio
    assert abs(ratio - generator.config.imbalance_ratio) < 0.05
    assert majority_count == len(df_base[df_base["target"] == 0])


def test_generate_concept_drift_data(generator, config):
    df_drift = generator.generate_concept_drift_data()

    assert len(df_drift) == config.holdout_samples
    assert "target" in df_drift.columns


def test_dataset_validation(generator, config):
    df_base = generator.generate_base_data()

    num_features = config.n_features - config.n_categorical

    # Validation should pass on correct data
    validated_df = validate_dataset(df_base, num_features, config.n_categorical)
    assert isinstance(validated_df, pd.DataFrame)

    # Should fail if we drop a required column
    df_invalid = df_base.drop(columns=["target"])
    with pytest.raises(pa.errors.SchemaError):
        validate_dataset(df_invalid, num_features, config.n_categorical)
