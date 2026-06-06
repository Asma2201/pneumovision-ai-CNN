import os
import torch
import torch.nn as nn
from model import create_model, get_device
from utils import get_dataloaders
from tqdm import tqdm

# 1. CONFIGURATION
CONFIG = {
    "data_dir"   : "data",
    "batch_size" : 32,
    "epochs"     : 10,
    "lr"         : 0.001,
    "model_path" : "models/model.pth"
}
# 2. ENTRAINEMENT
def train(model, train_loader, val_loader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=CONFIG["lr"])

    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):

        # ── Phase entraînement ──
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Phase validation ──
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Métriques ──
        train_acc = train_correct / len(train_loader.dataset)
        val_acc   = val_correct   / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} "
              f"| Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} "
              f"| Val Acc: {val_acc:.4f}")

        # ── Sauvegarde du meilleur modèle ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"  ✓ Meilleur modèle sauvegardé (val_acc: {val_acc:.4f})")

# 3. LANCEMENT
if __name__ == "__main__":
    device                        = get_device()
    train_loader, val_loader, _   = get_dataloaders(CONFIG["data_dir"], CONFIG["batch_size"])
    model                         = create_model().to(device)

    os.makedirs("models", exist_ok=True)
    train(model, train_loader, val_loader, device)