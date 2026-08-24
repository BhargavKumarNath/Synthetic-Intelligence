import logging

import h2o
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OracleSampler:
    """
    Model-driven rejection sampling.
    Generates synthetic samples by interpolating minority class samples (similar to SMOTE),
    then uses a trained 'Oracle' model to evaluate the confidence of those samples.
    Only samples that the model confidently predicts as the minority class are kept.
    """

    def __init__(
        self,
        oracle_model_path: str,
        confidence_threshold: float = 0.75,
        batch_size: int = 5000,
        max_attempts: int = 1000000,
        random_state: int = 42,
    ):

        self.oracle_model_path = oracle_model_path
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.random_state = random_state
        self.oracle_model = None

    def _ensure_h2o_and_model(self):
        if self.oracle_model is None:
            try:
                if h2o.cluster() is None:
                    h2o.init(nthreads=-1, max_mem_size="12g")
            except Exception:
                h2o.init(nthreads=-1, max_mem_size="12g")
            logger.info(f"Loading Oracle model from {self.oracle_model_path}")
            self.oracle_model = h2o.load_model(self.oracle_model_path)

    def _generate_batch(
        self,
        minority_data_num: np.ndarray,
        minority_data_cat: np.ndarray,
        numerical_cols: list,
        categorical_cols: list,
    ) -> list[dict]:

        batch_samples = []
        n_minority = (
            len(minority_data_num)
            if len(minority_data_num) > 0
            else len(minority_data_cat)
        )

        if n_minority >= 2:
            p1_indices = np.random.choice(n_minority, self.batch_size, replace=True)
            p2_indices = np.random.choice(n_minority, self.batch_size, replace=True)

            mask = p1_indices == p2_indices
            while mask.any():
                p2_indices[mask] = np.random.choice(
                    n_minority, mask.sum(), replace=True
                )
                mask = p1_indices == p2_indices

            interpolation_ratios = np.random.rand(self.batch_size)
            categorical_choices = np.random.rand(self.batch_size) > 0.5

            for i in range(self.batch_size):
                new_sample = {}

                if len(numerical_cols) > 0:
                    p1_num = minority_data_num[p1_indices[i]]
                    p2_num = minority_data_num[p2_indices[i]]
                    interpolated = p1_num * interpolation_ratios[i] + p2_num * (
                        1 - interpolation_ratios[i]
                    )
                    new_sample.update(dict(zip(numerical_cols, interpolated)))

                if len(categorical_cols) > 0:
                    cat_values = (
                        minority_data_cat[p1_indices[i]]
                        if categorical_choices[i]
                        else minority_data_cat[p2_indices[i]]
                    )
                    new_sample.update(dict(zip(categorical_cols, cat_values)))

                batch_samples.append(new_sample)

        return batch_samples

    def sample(
        self, df: pd.DataFrame, n_samples_to_generate: int, target_col: str = "target"
    ) -> pd.DataFrame:
        self._ensure_h2o_and_model()

        df_minority = df[df[target_col] == 1].drop(columns=[target_col])
        df_majority = df[df[target_col] == 0]

        numerical_cols = df_minority.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_minority.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        minority_numerical = (
            df_minority[numerical_cols].values if numerical_cols else np.array([])
        )
        minority_categorical = (
            df_minority[categorical_cols].values if categorical_cols else np.array([])
        )

        predictors = df_minority.columns.tolist()

        np.random.seed(self.random_state)
        synthetic_samples = []
        total_attempts = 0

        logger.info(
            f"Starting Oracle-driven generation for {n_samples_to_generate} samples..."
        )

        while (
            len(synthetic_samples) < n_samples_to_generate
            and total_attempts < self.max_attempts
        ):
            batch_samples = self._generate_batch(
                minority_numerical,
                minority_categorical,
                numerical_cols,
                categorical_cols,
            )

            if not batch_samples:
                break

            batch_df = pd.DataFrame(batch_samples)
            batch_df = batch_df.reindex(columns=predictors, fill_value=0)

            try:
                batch_hf = h2o.H2OFrame(batch_df)
                batch_predictions = self.oracle_model.predict(batch_hf)
                batch_predictions_df = batch_predictions.as_data_frame()

                # 'p1' is the probability of class 1 in H2O classification output
                confident_mask = batch_predictions_df["p1"] >= self.confidence_threshold
                confident_samples = batch_df[confident_mask].copy()

                if len(confident_samples) > 0:
                    confident_samples[target_col] = 1
                    synthetic_samples.extend(confident_samples.to_dict("records"))

                total_attempts += self.batch_size

                if total_attempts % (self.batch_size * 2) == 0:
                    acceptance_rate = len(synthetic_samples) / total_attempts * 100
                    logger.info(
                        f"Attempts: {total_attempts}, Generated: {len(synthetic_samples)}/{n_samples_to_generate} (Acceptance rate: {acceptance_rate:.2f}%)"
                    )

            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                continue

        if len(synthetic_samples) > n_samples_to_generate:
            synthetic_samples = synthetic_samples[:n_samples_to_generate]

        logger.info(
            f"Generation completed. Generated {len(synthetic_samples)} samples out of {total_attempts} attempts."
        )

        if synthetic_samples:
            df_synthetic = pd.DataFrame(synthetic_samples)
            df_final = pd.concat(
                [df_majority, df_minority.assign(**{target_col: 1}), df_synthetic],
                ignore_index=True,
            )
            df_final = df_final.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
            return df_final
        else:
            logger.warning(
                "No synthetic samples generated. Check your confidence threshold."
            )
            return df
