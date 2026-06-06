import torch
from torchvision import transforms
from PIL import Image
from model import create_model, get_device


# ─────────────────────────────────────────
# 1. TRANSFORMATIONS
# ─────────────────────────────────────────

predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. CHARGEMENT DU MODELE

def load_model(model_path, device):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 3. PREDICTION

def predict(image_path, model, device):

    # Charger et transformer l'image
    image  = Image.open(image_path).convert("RGB")
    tensor = predict_transforms(image).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        outputs      = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence,  predicted = torch.max(probabilities, 1)

    # Classes
    classes = ["NORMAL", "PNEUMONIA"]
    label   = classes[predicted.item()]
    score   = confidence.item() * 100

    return label, score


# 4. TEST RAPIDE


if __name__ == "__main__":
    import sys

    device     = get_device()
    model      = load_model("models/model.pth", device)

    # Passe le chemin d'une image en argument
    image_path = sys.argv[1]
    label, score = predict(image_path, model, device)

    print(f"Résultat    : {label}")
    print(f"Confiance   : {score:.2f}%")