import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os

class CrackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

sd = torch.load('model/best_crack_model.pt', map_location='cpu', weights_only=False)
model = CrackCNN()
model.load_state_dict(sd)
model.eval()

# Generate synthetic test images to understand class orientation
# Create a clearly cracked image (black line on gray)
# Create a clearly clean image (uniform gray)

def predict_all_modes(img_pil, name):
    print("--- %s ---" % name)
    arr = np.array(img_pil.resize((64,64)).convert('RGB'))
    print("  Pixel stats: mean=%.1f  std=%.1f  min=%d  max=%d" % (
        arr.mean(), arr.std(), arr.min(), arr.max()))
    
    transforms_to_try = [
        ("No norm  (/255)",          transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])),
        ("ImageNet norm",            transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), 
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])),
        ("0.5/0.5 norm",             transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
                                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])),
        ("Grayscale /255",           transforms.Compose([transforms.Grayscale(3), transforms.Resize((64,64)), transforms.ToTensor()])),
    ]
    
    with torch.no_grad():
        for tname, tf in transforms_to_try:
            t = tf(img_pil).unsqueeze(0)
            raw = model(t).item()
            prob = torch.sigmoid(torch.tensor(raw)).item()
            # Both interpretations
            if prob > 0.5:
                interp1 = "CRACK (%.0f%%)" % (prob*100)
                interp2 = "NO CRACK (inverted, %.0f%%)" % (prob*100)
            else:
                interp1 = "NO CRACK (%.0f%%)" % ((1-prob)*100)
                interp2 = "CRACK (inverted, %.0f%%)" % ((1-prob)*100)
            print("  %-22s raw=%6.2f  prob=%.3f  ->  %s" % (tname, raw, prob, interp1))

# Synthetic: clean concrete (uniform gray ~180)
clean = Image.fromarray(np.full((224,224,3), 180, dtype=np.uint8))
predict_all_modes(clean, "SYNTHETIC: Clean concrete (uniform gray 180)")

# Synthetic: cracked (gray with black diagonal line)
cracked_arr = np.full((224,224,3), 180, dtype=np.uint8)
for i in range(224):
    cracked_arr[i, i] = [0, 0, 0]
    if i > 0: cracked_arr[i, i-1] = [20, 20, 20]
    if i < 223: cracked_arr[i, i+1] = [20, 20, 20]
cracked = Image.fromarray(cracked_arr)
predict_all_modes(cracked, "SYNTHETIC: Cracked (gray + black diagonal line)")

# Synthetic: very dark (crack-like texture simulation)
dark_arr = np.random.randint(30, 80, (224,224,3), dtype=np.uint8)
dark = Image.fromarray(dark_arr)
predict_all_modes(dark, "SYNTHETIC: Dark noisy (crack-like)")

# Synthetic: bright clean
bright_arr = np.random.randint(180, 230, (224,224,3), dtype=np.uint8)
bright = Image.fromarray(bright_arr)
predict_all_modes(bright, "SYNTHETIC: Bright noisy (clean-like)")

print()
print("CONCLUSION: Look for which combo gives CRACK for the crack image and NO CRACK for clean.")
