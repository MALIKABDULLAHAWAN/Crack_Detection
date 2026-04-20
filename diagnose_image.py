"""
Final diagnosis: test the actual image file the user is uploading
through all normalization strategies and decide the correct one.
Run this script: python diagnose_image.py path/to/your/image.jpg
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

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

if len(sys.argv) < 2:
    print("Usage: python diagnose_image.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
print("Image:", img_path)
print("Size:", img.size)
print("Pixel stats: mean=%.1f  std=%.1f  min=%d  max=%d" % (
    arr.mean(), arr.std(), arr.min(), arr.max()))
print()

transforms_list = [
    ("No norm (/255 only)",    transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])),
    ("ImageNet normalize",     transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])),
    ("0.5/0.5 normalize",      transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
                                   transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])),
]

print("%-25s  %8s  %8s  %s" % ("Preprocessing", "raw", "prob", "prediction"))
print("-" * 65)
with torch.no_grad():
    for name, tf in transforms_list:
        t = tf(img).unsqueeze(0)
        t_stats = t.numpy()
        raw = model(t).item()
        prob = torch.sigmoid(torch.tensor(raw)).item()
        if prob > 0.5:
            pred = "CRACK DETECTED (%.1f%%)" % (prob*100)
        else:
            pred = "NO CRACK (%.1f%%)" % ((1-prob)*100)
        print("%-25s  %8.2f  %8.4f  %s" % (name, raw, prob, pred))
