from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from synthetic_intelligence.samplers import GraphSampler, OracleSampler, SmoteSampler


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    # 90 majority, 10 minority
    X_num = np.random.randn(100, 3)
    # mock categorical as numeric for SMOTE simplicity
    X_cat = np.random.randint(0, 2, (100, 2))
    X = np.hstack((X_num, X_cat))

    y = np.array([0] * 90 + [1] * 10)

    df = pd.DataFrame(X, columns=["num1", "num2", "num3", "cat1", "cat2"])
    df["target"] = y

    # SMOTE expects no categorical string columns out of the box unless specified,
    # so we just test with purely numeric representation.
    return df


def test_smote_sampler(dummy_data):
    sampler = SmoteSampler(random_state=42)

    X = dummy_data.drop(columns=["target"])
    y = dummy_data["target"]

    X_res, y_res = sampler.sample(X, y)

    # SMOTE should balance the classes
    assert len(y_res) == 180
    assert y_res.value_counts()[0] == 90
    assert y_res.value_counts()[1] == 90


def test_graph_sampler(dummy_data):
    # Ask for 50 synthetic samples
    sampler = GraphSampler(n_neighbors=3, batch_size=10, random_state=42)

    df_res = sampler.sample(dummy_data, n_samples_to_generate=50, target_col="target")

    # Original data: 100 rows. Augmented: 100 + 50 = 150 rows.
    assert len(df_res) == 150
    assert df_res["target"].value_counts()[1] == 10 + 50
    assert df_res["target"].value_counts()[0] == 90


@patch("synthetic_intelligence.samplers.oracle.h2o")
def test_oracle_sampler(mock_h2o, dummy_data):
    # Mock H2O framework
    mock_model = MagicMock()

    # Mock prediction: return a dataframe with a 'p1' column
    # Let's say all batches return p1=0.8 (above threshold 0.75)
    def mock_predict(hf):
        # hf is a mocked H2OFrame, we just need to return a mock with as_data_frame
        pred_df = pd.DataFrame(
            {"predict": [1] * len(hf), "p0": [0.2] * len(hf), "p1": [0.8] * len(hf)}
        )
        mock_pred = MagicMock()
        mock_pred.as_data_frame.return_value = pred_df
        return mock_pred

    mock_model.predict = mock_predict
    mock_h2o.load_model.return_value = mock_model

    # The h2o.H2OFrame constructor needs to just return a dummy length object
    # The OracleSampler passes a batch_df to it.
    mock_h2o.H2OFrame.side_effect = lambda df: df

    sampler = OracleSampler(
        oracle_model_path="dummy_path",
        confidence_threshold=0.75,
        batch_size=10,
        max_attempts=100,
    )

    df_res = sampler.sample(dummy_data, n_samples_to_generate=20, target_col="target")

    # Should have generated exactly 20 new samples
    assert len(df_res) == 120
    assert df_res["target"].value_counts()[1] == 30
    assert df_res["target"].value_counts()[0] == 90
