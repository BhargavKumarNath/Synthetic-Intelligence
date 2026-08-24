import logging
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GraphSampler:
    def __init__(
        self, n_neighbors: int = 5, batch_size: int = 10000, random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_knn_graph(
        self, df_minority_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Building kNN graph with {self.n_neighbors} neighbors...")
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1, algorithm="ball_tree", n_jobs=-1
        )
        nn.fit(df_minority_scaled)
        distances, indices = nn.kneighbors(df_minority_scaled)
        return distances, indices

    def _generate_synthetic_batch(
        self,
        df_minority: pd.DataFrame,
        indices: np.ndarray,
        numerical_cols: list,
        batch_size: int,
        random_state: int,
    ) -> pd.DataFrame:

        np.random.seed(random_state)

        minority_numerical = df_minority[numerical_cols].values
        minority_categorical = df_minority.drop(columns=numerical_cols)
        n_minority_samples = len(df_minority)

        random_node_indices = np.random.randint(0, n_minority_samples, size=batch_size)
        synthetic_numerical = np.zeros((batch_size, len(numerical_cols)))
        categorical_indices = np.zeros(batch_size, dtype=int)

        for i, node_idx in enumerate(random_node_indices):
            # Pick a random neighbor (excluding the point itself which is at index 0)
            neighbor_idx = np.random.choice(indices[node_idx][1:])
            interpolation_ratio = np.random.rand()

            synthetic_numerical[i] = minority_numerical[
                node_idx
            ] * interpolation_ratio + minority_numerical[neighbor_idx] * (
                1 - interpolation_ratio
            )
            categorical_indices[i] = node_idx

        synthetic_df = pd.DataFrame(synthetic_numerical, columns=numerical_cols)

        if len(minority_categorical.columns) > 0:
            categorical_data = minority_categorical.iloc[
                categorical_indices
            ].reset_index(drop=True)
            synthetic_df = pd.concat([synthetic_df, categorical_data], axis=1)

        synthetic_df["target"] = 1
        return synthetic_df

    def sample(
        self, df: pd.DataFrame, n_samples_to_generate: int, target_col: str = "target"
    ) -> pd.DataFrame:
        """
        Generates synthetic samples using kNN graph interpolation and appends them to the original dataframe.
        Returns the new augmented dataframe.
        """
        df_minority = df[df[target_col] == 1].drop(columns=[target_col])
        df_majority = df[df[target_col] == 0]

        numerical_cols = df_minority.select_dtypes(include=np.number).columns.tolist()

        scaler = StandardScaler()
        df_minority_scaled = scaler.fit_transform(df_minority[numerical_cols])

        distances, indices = self._build_knn_graph(df_minority_scaled)

        logger.info(
            f"Generating {n_samples_to_generate} synthetic samples in batches of {self.batch_size}..."
        )

        synthetic_batches = []
        n_batches = (n_samples_to_generate + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_time = time.time()
            current_batch_size = min(
                self.batch_size, n_samples_to_generate - batch_idx * self.batch_size
            )
            batch_random_state = self.random_state + batch_idx

            synthetic_batch = self._generate_synthetic_batch(
                df_minority,
                indices,
                numerical_cols,
                current_batch_size,
                batch_random_state,
            )
            synthetic_batches.append(synthetic_batch)
            logger.info(
                f"Batch {batch_idx + 1}/{n_batches} completed in {time.time() - start_time:.2f}s ({current_batch_size} samples)"
            )

        df_synthetic = pd.concat(synthetic_batches, ignore_index=True)
        # Reorder columns to match original df
        df_synthetic = df_synthetic[df.drop(columns=[target_col]).columns]
        df_synthetic[target_col] = 1

        df_final = pd.concat(
            [df_majority, df_minority.assign(**{target_col: 1}), df_synthetic],
            ignore_index=True,
        )
        df_final = df_final.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )

        return df_final
