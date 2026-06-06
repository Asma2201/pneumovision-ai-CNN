import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# 2. CHARGEMENT ET SPLIT DU DATASET
def get_dataloaders(data_dir, batch_size=32):

    # Chargement depuis les dossiers NORMAL/ et PNEUMONIA/
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_transforms
    )

    # Split automatique 80% train / 20% validation
    train_size = int(0.8 * len(train_dataset))
    val_size   = len(train_dataset) - train_size

    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    # DataLoaders — servent les images en batches au modèle
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 3. TEST RAPIDE
# ─────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="data")

    print(f"Train batches     : {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches      : {len(test_loader)}")

    # Vérification d'un batch
    images, labels = next(iter(train_loader))
    print(f"Shape d'un batch  : {images.shape}")
    print(f"Classes           : {train_loader.dataset.dataset.classes}")