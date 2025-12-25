import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog
from torchvision import transforms
import torch.nn.functional as F
# ---------- MODEL DEFINITION ----------
class LungCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------- LOAD MODEL (LOCAL PATH) ----------
MODEL_PATH = r"src\mode_detect_pnemoina\lung_cnn_model.pth"   # 👈 CHANGE IF NEEDEDexi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LungCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = ["NORMAL", "PNEUMONIA"]

# ---------- FILE BROWSER ----------
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select Chest X-ray Image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

# ---------- INFERENCE ----------
img = Image.open(image_path).convert("L")
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)

prediction = class_names[pred.item()]
confidence = conf.item() * 100

# ---------- DISPLAY ----------
plt.imshow(img, cmap="gray")
plt.title(f"{prediction} ({confidence:.2f}%)")
plt.axis("off")
plt.show()

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")
