import logging

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim_1 = max(256, input_dim // 2)
        hidden_dim_2 = max(128, input_dim // 4)
        hidden_dim_3 = max(64, latent_dim * 8)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.BatchNorm1d(hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_3, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_3),
            nn.BatchNorm1d(hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_3, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AutoencoderTrainer:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        learning_rate: float = 5e-4,
        experiment_name: str = "Autoencoder",
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)

    def train(self, X_train: torch.Tensor, epochs: int = 100, batch_size: int = 256):
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "input_dim": self.input_dim,
                    "latent_dim": self.latent_dim,
                    "learning_rate": self.learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                }
            )

            dataset = TensorDataset(X_train, X_train)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=10, factor=0.5
            )

            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for data, _ in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    reconstructed, _ = self.model(data)
                    loss = criterion(reconstructed, data)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                scheduler.step(avg_loss)

                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                if (epoch + 1) % 10 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {lr:.6f}"
                    )

            # Log the model to MLflow
            # Use pickle serialization to avoid TorchScript tracing issues with BatchNorm
            self.model.eval()
            mlflow.pytorch.log_model(
                self.model, "autoencoder_model", serialization_format="pickle"
            )
            logger.info("Training complete and model logged to MLflow.")

        return self.model
