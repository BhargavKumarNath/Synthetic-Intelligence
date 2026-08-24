import hydra
from omegaconf import DictConfig, OmegaConf
from synthetic_intelligence.config import ProjectConfig
from synthetic_intelligence.data.generator import SyntheticDatasetGenerator
from synthetic_intelligence.data.validator import validate_dataset

@hydra.main(version_base=None, config_path="conf", config_name="config")
def smoke_test(cfg: DictConfig):
    print("=== Phase 1 Smoke Test ===")
    
    # 1. Parse Config to Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    project_config = ProjectConfig(**cfg_dict)
    gen_config = project_config.generation
    print(f"Loaded config successfully. n_samples={gen_config.n_samples}")
    
    # 2. Instantiate Generator
    generator = SyntheticDatasetGenerator(gen_config)
    
    # 3. Generate Data
    print("Generating base data...")
    df_base = generator.generate_base_data()
    print(f"Base data generated. Shape: {df_base.shape}")
    
    print("Generating imbalanced data...")
    df_imbalanced = generator.create_imbalanced_dataset(df_base)
    print(f"Imbalanced data generated. Shape: {df_imbalanced.shape}")
    
    print("Generating concept drift data...")
    df_drift = generator.generate_concept_drift_data()
    print(f"Concept drift data generated. Shape: {df_drift.shape}")
    
    # 4. Validate Data
    print("Validating datasets with Pandera...")
    num_features = gen_config.n_features - gen_config.n_categorical
    validate_dataset(df_base, num_features, gen_config.n_categorical)
    validate_dataset(df_imbalanced, num_features, gen_config.n_categorical)
    validate_dataset(df_drift, num_features, gen_config.n_categorical)
    print("All datasets passed Pandera validation!")
    
    print("=== Phase 1 Smoke Test Passed! ===")

if __name__ == "__main__":
    smoke_test()
