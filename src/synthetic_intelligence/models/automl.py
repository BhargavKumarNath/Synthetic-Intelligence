import logging

import h2o
import mlflow
import mlflow.h2o
from h2o.automl import H2OAutoML

logger = logging.getLogger(__name__)


class AutoMLTrainer:
    def __init__(
        self,
        max_runtime_secs: int = 3600,
        max_models: int = 20,
        seed: int = 42,
        experiment_name: str = "H2O_AutoML",
    ):
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.seed = seed
        self.experiment_name = experiment_name

    def _ensure_h2o(self):
        try:
            if h2o.cluster() is None:
                h2o.init(nthreads=-1, max_mem_size="12g")
        except Exception:
            h2o.init(nthreads=-1, max_mem_size="12g")

    def train(
        self,
        df_train,
        df_valid,
        predictors: list,
        response: str,
        run_name: str = "automl_run",
    ):
        self._ensure_h2o()

        hf_train = h2o.H2OFrame(df_train)
        hf_valid = h2o.H2OFrame(df_valid)

        hf_train[response] = hf_train[response].asfactor()
        hf_valid[response] = hf_valid[response].asfactor()

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "max_runtime_secs": self.max_runtime_secs,
                    "max_models": self.max_models,
                    "seed": self.seed,
                    "predictors_count": len(predictors),
                }
            )

            logger.info(f"Starting H2O AutoML for {run_name}...")
            aml = H2OAutoML(
                max_runtime_secs=self.max_runtime_secs,
                max_models=self.max_models,
                seed=self.seed,
                sort_metric="AUCPR",
            )

            aml.train(
                x=predictors,
                y=response,
                training_frame=hf_train,
                validation_frame=hf_valid,
            )

            leader_model = aml.leader
            if leader_model is None:
                logger.error("AutoML failed to find a leader model.")
                raise ValueError("No models trained by H2O AutoML.")

            logger.info(f"Leader Model: {leader_model.model_id}")

            # Log metrics from the validation set
            perf = leader_model.model_performance(hf_valid)
            metrics = {
                "auc": perf.auc(),
                "aucpr": perf.aucpr(),
                "logloss": perf.logloss(),
                "rmse": perf.rmse(),
            }
            mlflow.log_metrics(metrics)

            # Log the model to MLflow
            mlflow.h2o.log_model(leader_model, "automl_leader_model")

            return aml
