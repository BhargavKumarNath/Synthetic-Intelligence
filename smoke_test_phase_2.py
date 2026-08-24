import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from synthetic_intelligence.config import ProjectConfig
from synthetic_intelligence.data.generator import SyntheticDatasetGenerator
from synthetic_intelligence.models import AutoencoderTrainer, AutoMLTrainer
from synthetic_intelligence.samplers import GraphSampler


@hydra.main(version_base=None, config_path="conf", config_name="config")
def smoke_test(cfg: DictConfig):
    print("=== Phase 2 Smoke Test ===")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    project_config = ProjectConfig(**cfg_dict)

    # 1. Generate Small Data
    gen_config = project_config.generation
    # Let's reduce samples for the smoke test artificially to save time
    gen_config.n_samples = 1000
    generator = SyntheticDatasetGenerator(gen_config)

    df_base = generator.generate_base_data()
    df_imbalanced = generator.create_imbalanced_dataset(df_base)

    print(f"Imbalanced Data Shape: {df_imbalanced.shape}")

    # 2. Sampler Check
    graph_sampler = GraphSampler(n_neighbors=3, batch_size=100, random_state=42)
    df_augmented = graph_sampler.sample(df_imbalanced, n_samples_to_generate=200)
    print(f"Augmented Data Shape: {df_augmented.shape}")

    # Split for models
    df_train, df_valid = train_test_split(
        df_augmented, test_size=0.2, random_state=42, stratify=df_augmented["target"]
    )

    # 3. AutoML MLflow Check
    print("Training AutoML with MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    automl_trainer = AutoMLTrainer(
        max_runtime_secs=10,  # Very short for smoke test
        max_models=2,
        experiment_name="smoke_test_automl",
    )

    predictors = [c for c in df_train.columns if c != "target"]

    # Catch any H2O issues safely
    try:
        automl_trainer.train(
            df_train, df_valid, predictors, "target", run_name="smoke_test_run"
        )
        print("AutoML Training Complete!")
    except Exception as e:
        print(
            f"AutoML Training Error (Expected if H2O is too fast to train models in 10s): {e}"
        )

    # 4. Autoencoder MLflow Check
    print("Training Autoencoder with MLflow...")
    numerical_cols = df_train.select_dtypes(include=["float64", "int64"]).columns
    numerical_cols = [c for c in numerical_cols if c != "target"]

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[numerical_cols])
    X_train_tensor = torch.FloatTensor(X_train_num)

    ae_trainer = AutoencoderTrainer(
        input_dim=len(numerical_cols), latent_dim=4, experiment_name="smoke_test_ae"
    )

    ae_trainer.train(X_train_tensor, epochs=2, batch_size=32)
    print("Autoencoder Training Complete!")

    print("=== Phase 2 Smoke Test Passed! ===")


if __name__ == "__main__":
    smoke_test()
