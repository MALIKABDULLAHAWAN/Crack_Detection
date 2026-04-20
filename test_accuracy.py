"""
SDNET2018 Accuracy Test
Tests the CNN model on actual training dataset images to find the
best preprocessing pipeline and measure real accuracy.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import os, random

DATASET_BASE = r"E:\FYP\DATA_Maguire_20180517_ALL\DATA_Maguire_20180517_ALL\SDNET2018"

# All cracked folders (C prefix) and uncracked (U prefix)
CRACK_FOLDERS = [
    os.path.join(DATASET_BASE, "D", "CD"),
    os.path.join(DATASET_BASE, "W", "CW"),
    os.path.join(DATASET_BASE, "P", "CP"),
]
NOCRACK_FOLDERS = [
    os.path.join(DATASET_BASE, "D", "UD"),
    os.path.join(DATASET_BASE, "W", "UW"),
    os.path.join(DATASET_BASE, "P", "UP"),
]

class CrackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

sd = torch.load('model/best_crack_model.pt', map_location='cpu', weights_only=False)
model = CrackCNN()
model.load_state_dict(sd)
model.eval()

# --- Preprocessing strategies ---
strategies = {
    "A: /255 only":      lambda img: torch.from_numpy(np.array(img.resize((64,64)),dtype=np.float32).transpose(2,0,1)/255.0).unsqueeze(0),
    "B: blur+/255":      lambda img: torch.from_numpy(np.array(img.resize((64,64)).filter(ImageFilter.GaussianBlur(0.5)),dtype=np.float32).transpose(2,0,1)/255.0).unsqueeze(0),
    "C: ImageNet norm":  lambda img: transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(img).unsqueeze(0),
}

def load_images_from_folders(folders, n=100):
    images = []
    for folder in folders:
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)
        for f in files[:n//len(folders)]:
            try:
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                images.append(img)
            except:
                pass
    return images

random.seed(42)
print("Loading images from SDNET2018...")
crack_imgs   = load_images_from_folders(CRACK_FOLDERS,   n=150)
nocrack_imgs = load_images_from_folders(NOCRACK_FOLDERS, n=150)
print(f"  Cracked images:     {len(crack_imgs)}")
print(f"  No-crack images:    {len(nocrack_imgs)}")
print()

print("%-22s  %8s  %8s  %8s  %8s" % ("Strategy", "Acc%", "CrackTP%", "NoCrackTN%", "Threshold"))
print("-" * 70)

best_acc = 0
best_strategy = None
best_threshold = 0.5

for name, tf in strategies.items():
    for threshold in [0.5, 0.4, 0.3, 0.6, 0.7]:
        correct_crack = 0
        correct_nocrack = 0

        with torch.no_grad():
            for img in crack_imgs:
                t = tf(img)
                p = torch.sigmoid(model(t)).item()
                if p > threshold:
                    correct_crack += 1

            for img in nocrack_imgs:
                t = tf(img)
                p = torch.sigmoid(model(t)).item()
                if p <= threshold:
                    correct_nocrack += 1

        total = len(crack_imgs) + len(nocrack_imgs)
        acc = (correct_crack + correct_nocrack) / total * 100
        tp  = correct_crack / len(crack_imgs) * 100
        tn  = correct_nocrack / len(nocrack_imgs) * 100

        marker = " <<< BEST" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_strategy = name
            best_threshold = threshold

        if threshold == 0.5:  # print all thresholds only for first time, else just 0.5
            print("%-22s  %7.1f%%  %7.1f%%  %9.1f%%  %6.2f%s" % (name, acc, tp, tn, threshold, marker))

print()
print("=" * 70)
print("BEST STRATEGY:  %s" % best_strategy)
print("BEST THRESHOLD: %.2f" % best_threshold)
print("BEST ACCURACY:  %.1f%%" % best_acc)
print("=" * 70)
print()

# Now run the best strategy at all thresholds to show the curve
print("\nTHRESHOLD SWEEP for best strategy (%s):" % best_strategy)
tf_best = strategies[best_strategy]
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    cc = cn = 0
    with torch.no_grad():
        for img in crack_imgs:
            if torch.sigmoid(model(tf_best(img))).item() > threshold: cc += 1
        for img in nocrack_imgs:
            if torch.sigmoid(model(tf_best(img))).item() <= threshold: cn += 1
    acc = (cc + cn) / (len(crack_imgs)+len(nocrack_imgs)) * 100
    print("  threshold=%.1f  acc=%.1f%%  crack_recall=%.1f%%  nocrack_recall=%.1f%%" % (
        threshold, acc, cc/len(crack_imgs)*100, cn/len(nocrack_imgs)*100))
