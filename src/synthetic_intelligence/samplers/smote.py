from __future__ import annotations

import logging

import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class SmoteSampler:
    def __init__(self, random_state: int = 42, sampling_strategy: str | float = "auto"):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.sampler = SMOTE(
            random_state=self.random_state, sampling_strategy=self.sampling_strategy
        )

    def sample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE to the dataset.
        Assumes categorical features have been encoded as numeric if present.
        """
        X_res, y_res = self.sampler.fit_resample(X, y)
        return X_res, y_res
