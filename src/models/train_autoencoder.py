"""
train_autoencoder.py — Treina Autoencoder com transacoes legitimas.
Uso: python src/models/train_autoencoder.py
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=1)

def train(epochs=50, batch_size=256, lr=1e-3, patience=7, device="cpu"):
    print("=" * 55)
    print("  Treinando Autoencoder - Anomaly Detection")
    print("=" * 55)

    X_legit = np.load(PROCESSED_DIR / "X_train_legit.npy").astype(np.float32)
    X_test  = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    val_size = int(0.2 * len(X_legit))
    X_val, X_tr = X_legit[:val_size], X_legit[val_size:]
    print(f"  Treino: {len(X_tr):,} | Validacao: {len(X_val):,} | Device: {device}")

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=batch_size, shuffle=True)
    val_tensor  = torch.tensor(X_val).to(device)
    test_tensor = torch.tensor(X_test).to(device)

    model = FraudAutoencoder(input_dim=X_legit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss, patience_counter, epoch = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), MODELS_DIR / "autoencoder_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping na epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODELS_DIR / "autoencoder_best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        test_errors = model.reconstruction_error(test_tensor).numpy()

    threshold = float(np.percentile(test_errors[y_test == 0], 95))
    print(f"\n  Threshold (p95): {threshold:.6f}")

    from sklearn.metrics import classification_report, average_precision_score
    ap = average_precision_score(y_test, test_errors)
    print(f"  PR-AUC: {ap:.4f}")
    print(classification_report(y_test, (test_errors > threshold).astype(int),
                                target_names=["Legitima", "Fraude"]))

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    json.dump({"model": "FraudAutoencoder", "input_dim": int(X_legit.shape[1]),
               "threshold": threshold, "pr_auc": round(ap, 4),
               "best_val_loss": round(best_val_loss, 6), "epochs_trained": epoch},
              open(MODELS_DIR / "autoencoder_metadata.json", "w"), indent=2)

    print(f"\n  Modelo salvo: {MODELS_DIR / 'autoencoder.pt'}")
    print("=" * 55)
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)
