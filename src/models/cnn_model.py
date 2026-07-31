import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from src.models.base import BaseModel
from src import config

class APUSlidingWindowDataset(Dataset):
    """
    Dataset to build sliding windows for the 1D-CNN model.
    Given a feature matrix of shape (N, D), it yields windows of shape (D, seq_len)
    along with labels of shape (3,) for the 2h, 4h, and 8h horizons.
    """
    def __init__(self, X: np.ndarray, y_2h: np.ndarray, y_4h: np.ndarray, y_8h: np.ndarray, seq_len: int = 60):
        self.X = X.astype(np.float32)
        # Target labels shape (N, 3)
        self.y = np.stack([y_2h, y_4h, y_8h], axis=1).astype(np.float32)
        self.seq_len = seq_len
        
    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Window of shape (seq_len, num_features)
        x_window = self.X[idx : idx + self.seq_len]
        # Transpose to shape (num_features, seq_len) for Conv1D
        x_tensor = torch.tensor(x_window).t()
        
        # Label is the target at the end of the sliding window
        y_label = torch.tensor(self.y[idx + self.seq_len - 1])
        return x_tensor, y_label

class CNN1DNet(nn.Module):
    """
    1D CNN architecture for time series classification.
    Input shape: (batch_size, num_features, seq_len)
    Output shape: (batch_size, 3) -> representing logits for 2h, 4h, 8h horizons.
    """
    def __init__(self, num_features: int, seq_len: int = 60):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 3) # 3 outputs for the three horizons
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x) # shape: (batch_size, 128, 1)
        
        x = x.squeeze(-1) # shape: (batch_size, 128)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc_out(x) # shape: (batch_size, 3)
        return logits

class APUPredictiveLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module to handle training, validation, losses, and learning rates.
    """
    def __init__(self, num_features: int, seq_len: int = 60, lr: float = 1e-3, pos_weights: List[float] = None):
        super().__init__()
        self.save_hyperparameters()
        self.net = CNN1DNet(num_features, seq_len)
        self.lr = lr
        
        # Setup pos weights for class imbalance
        if pos_weights is not None:
            self.register_buffer("pos_weights", torch.tensor(pos_weights, dtype=torch.float32))
        else:
            self.register_buffer("pos_weights", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes BCE loss ignoring neutral/ignore labels (-1).
        """
        loss = 0.0
        valid_counts = 0
        
        for i in range(3):
            # Target labels for horizon i
            t_i = targets[:, i]
            # Logits for horizon i
            l_i = logits[:, i]
            
            # Mask out neutral samples (-1)
            mask = (t_i != -1)
            if mask.sum() == 0:
                continue
                
            # Apply class weights
            weight = self.pos_weights[i]
            loss_i = F.binary_cross_entropy_with_logits(
                l_i[mask], 
                t_i[mask], 
                pos_weight=weight
            )
            loss += loss_i
            valid_counts += 1
            
        return loss / max(1, valid_counts)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class CNN1DModel(BaseModel):
    """
    1D-CNN wrapper model satisfying the BaseModel interface.
    """
    def __init__(self, num_features: int, seq_len: int = 60, lr: float = 1e-3, epochs: int = 10, batch_size: int = 256):
        self.num_features = num_features
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lightning_module = None
        
    def fit(self, X, y, **kwargs) -> "CNN1DModel":
        # Extract individual label targets
        # y should be a dict or a 2D numpy array where:
        # col 0: 2h labels, col 1: 4h labels, col 2: 8h labels
        y_2h = y[:, 0]
        y_4h = y[:, 1]
        y_8h = y[:, 2]
        
        # Calculate pos weights
        pos_weights = []
        for y_h in [y_2h, y_4h, y_8h]:
            clean_y = y_h[y_h != -1]
            n_neg = np.sum(clean_y == 0)
            n_pos = np.sum(clean_y == 1)
            weight = n_neg / max(1, n_pos)
            pos_weights.append(weight)
            
        # Create DataLoaders
        train_dataset = APUSlidingWindowDataset(X, y_2h, y_4h, y_8h, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        self.lightning_module = APUPredictiveLightningModule(
            num_features=self.num_features,
            seq_len=self.seq_len,
            lr=self.lr,
            pos_weights=pos_weights
        )
        
        # Run Trainer
        # Automatically select GPU if available
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False
        )
        
        trainer.fit(self.lightning_module, train_loader)
        return self
        
    def predict_proba(self, X) -> np.ndarray:
        """
        Returns probabilities of shape (N, 3) for the 3 horizons.
        """
        if self.lightning_module is None:
            raise ValueError("Model is not fitted yet.")
            
        self.lightning_module.eval()
        # Feed all window slices
        # We can construct window loader for inference
        # To make it fast, we can use sliding window tensor operations in PyTorch
        probs_list = []
        
        with torch.no_grad():
            # For inference, create non-shuffled loader
            dataset = APUSlidingWindowDataset(X, np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X)), self.seq_len)
            loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)
            
            device = self.lightning_module.device
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                logits = self.lightning_module(batch_x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())
                
        probs_all = np.vstack(probs_list)
        
        # Pad the beginning with zeros since sliding window starts at seq_len - 1
        pad_size = len(X) - len(probs_all)
        if pad_size > 0:
            padding = np.zeros((pad_size, 3))
            probs_all = np.vstack([padding, probs_all])
            
        return probs_all
        
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save model state dict + structure params
        save_dict = {
            "num_features": self.num_features,
            "seq_len": self.seq_len,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "state_dict": self.lightning_module.state_dict() if self.lightning_module else None
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
            
    def load(self, path: Path) -> "CNN1DModel":
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
            
        self.num_features = save_dict["num_features"]
        self.seq_len = save_dict["seq_len"]
        self.lr = save_dict["lr"]
        self.epochs = save_dict["epochs"]
        self.batch_size = save_dict["batch_size"]
        
        if save_dict["state_dict"] is not None:
            self.lightning_module = APUPredictiveLightningModule(
                num_features=self.num_features,
                seq_len=self.seq_len,
                lr=self.lr
            )
            self.lightning_module.load_state_dict(save_dict["state_dict"])
            
        return self
