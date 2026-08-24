import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from synthetic_intelligence.config import GenerationConfig

class SyntheticDatasetGenerator:
    """
    Generates synthetic tabular data for classification tasks.
    Mimics a complex fraud/risk detection scenario with numerical and categorical features.
    """
    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate_base_data(self) -> pd.DataFrame:
        """
        Generates the core balanced numerical dataset and adds categorical features.
        Returns:
            pd.DataFrame: A balanced dataset with numerical and categorical features.
        """
        print("Generating core numerical features...")
        num_features = self.config.n_features - self.config.n_categorical
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=num_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_classes=2,
            class_sep=self.config.class_sep,
            flip_y=self.config.flip_y,
            weights=[0.5, 0.5],
            random_state=self.config.random_state
        )

        df_numerical = pd.DataFrame(X, columns=[f'num_{i}' for i in range(X.shape[1])])
        df_target = pd.DataFrame(y, columns=['target'])

        print("Generating categorical features...")
        df_categorical = pd.DataFrame()
        # Set a seed for reproducibility of categorical generation
        np.random.seed(self.config.random_state) 
        
        for i in range(self.config.n_categorical):
            # Random cardinality between 3 and 15
            num_categories = np.random.randint(3, 15) 
            categories = [f'cat_{i}_val_{j}' for j in range(num_categories)]
            
            cat_data = np.random.choice(categories, size=self.config.n_samples)
            df_categorical[f'cat_{i}'] = pd.Series(cat_data, dtype='category')

        df_balanced = pd.concat([df_numerical, df_categorical, df_target], axis=1)
        return df_balanced

    def create_imbalanced_dataset(self, df_balanced: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a realistic imbalanced dataset from the balanced ground truth.
        """
        df_majority = df_balanced[df_balanced['target'] == 0]
        df_minority = df_balanced[df_balanced['target'] == 1]

        # Calculate needed minority samples for the specific ratio
        n_minority_new = int(len(df_majority) * self.config.imbalance_ratio / (1 - self.config.imbalance_ratio))
        
        df_minority_sampled = df_minority.sample(n=n_minority_new, random_state=self.config.random_state)
        
        df_imbalanced = pd.concat([df_majority, df_minority_sampled]).sample(frac=1, random_state=self.config.random_state)
        return df_imbalanced

    def generate_concept_drift_data(self) -> pd.DataFrame:
        """
        Generates a 'New World' dataset with shifted distributions to test generalization.
        """
        # We simulate drift by changing the random state and slightly modifying parameters
        print("Generating concept drift data...")
        num_features = self.config.n_features - self.config.n_categorical
        X, y = make_classification(
            n_samples=self.config.holdout_samples,
            n_features=num_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_classes=2,
            class_sep=self.config.class_sep * 0.9, # Slightly harder covariance
            flip_y=self.config.flip_y + 0.02,     # More noise
            weights=[0.5, 0.5],
            random_state=self.config.new_world_random_state # Drifted Seed
        )

        df_numerical = pd.DataFrame(X, columns=[f'num_{i}' for i in range(X.shape[1])])
        df_target = pd.DataFrame(y, columns=['target'])
        
        # Categorical drift: maybe different proportions or same logic with new seed
        df_categorical = pd.DataFrame()
        np.random.seed(self.config.new_world_random_state)
        
        for i in range(self.config.n_categorical):
            num_categories = np.random.randint(3, 15)
            categories = [f'cat_{i}_val_{j}' for j in range(num_categories)]
            cat_data = np.random.choice(categories, size=self.config.holdout_samples)
            df_categorical[f'cat_{i}'] = pd.Series(cat_data, dtype='category')

        df_drifted = pd.concat([df_numerical, df_categorical, df_target], axis=1)
        return df_drifted
