"""
Comprehensive preprocessing finder.
Tests 6 different preprocessing strategies against synthetic images
to find which one produces physically sensible results.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np

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
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

sd = torch.load('model/best_crack_model.pt', map_location='cpu', weights_only=False)
model = CrackCNN()
model.load_state_dict(sd)
model.eval()

def make_tensor(pil_img, strategy):
    img64 = pil_img.resize((64,64))
    if strategy == 'A':  # /255 no norm
        arr = np.array(img64, dtype=np.float32)/255.0
        return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    elif strategy == 'B':  # ImageNet norm
        tf = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        return tf(img64).unsqueeze(0)
    elif strategy == 'C':  # blur + /255
        img64 = img64.filter(ImageFilter.GaussianBlur(0.8))
        arr = np.array(img64, dtype=np.float32)/255.0
        return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    elif strategy == 'D':  # /255 + per-image normalize
        arr = np.array(img64, dtype=np.float32)/255.0
        mean = arr.mean(axis=(0,1))
        std = arr.std(axis=(0,1)) + 1e-8
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    elif strategy == 'E':  # grayscale to RGB /255
        gray = img64.convert('L').convert('RGB')
        arr = np.array(gray, dtype=np.float32)/255.0
        return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)

def run(img, name, strategy):
    with torch.no_grad():
        t = make_tensor(img, strategy)
        p = torch.sigmoid(model(t)).item()
        return p

# === CREATE TEST IMAGES ===
np.random.seed(42)

# 1. Perfect clean concrete (bright, uniform, slight grain)
clean_arr = np.random.normal(185, 8, (64,64,3)).clip(0,255).astype(np.uint8)
clean = Image.fromarray(clean_arr)

# 2. Concrete with a clear crack (dark thin line)
crack_arr = np.random.normal(185, 8, (64,64,3)).clip(0,255).astype(np.uint8)
# Draw a crack: dark diagonal line
for i in range(10, 55):
    for d in [-1, 0, 1]:
        if 0 <= i+d < 64:
            crack_arr[i, i+d] = np.array([15, 15, 15])
crack = Image.fromarray(crack_arr)

# 3. Real-world concrete: medium texture (staining, lighting variation)
real_arr = np.random.normal(170, 28, (64,64,3)).clip(0,255).astype(np.uint8)
real = Image.fromarray(real_arr)

strategies = ['A', 'B', 'C', 'D', 'E']
names = ['A:/255 only', 'B:ImageNet', 'C:blur+/255', 'D:per-img-norm', 'E:grayscale']

print("STRATEGY | Clean(no crack) | Clear Crack | Real Texture | CORRECT?")
print("-" * 75)
for s, sname in zip(strategies, names):
    p_clean = run(clean, 'clean', s)
    p_crack = run(crack, 'crack', s)
    p_real  = run(real,  'real',  s)
    
    # What we expect: clean=NO CRACK (<0.5), crack=CRACK (>0.5), real=NO CRACK (<0.5)
    clean_ok = 'OK' if p_clean < 0.5 else 'WRONG'
    crack_ok = 'OK' if p_crack > 0.5 else 'WRONG'
    real_ok  = 'OK' if p_real < 0.5 else 'WRONG (false alarm)'
    
    all_ok = 'YES' if (clean_ok == 'OK' and crack_ok == 'OK' and real_ok == 'OK') else 'NO'
    
    print("%-14s | clean=%.3f %-6s | crack=%.3f %-6s | real=%.3f %-18s | %s" % (
        sname, p_clean, clean_ok, p_crack, crack_ok, p_real, real_ok, all_ok))

print()
print("ALSO TESTING: What if label is inverted? (0=crack, 1=no_crack)")
print("-" * 75)
for s, sname in zip(strategies, names):
    p_clean = run(clean, 'clean', s)
    p_crack = run(crack, 'crack', s)
    p_real  = run(real,  'real',  s)
    
    # Inverted: crack = sigmoid < 0.5
    clean_ok = 'OK' if p_clean > 0.5 else 'WRONG'
    crack_ok = 'OK' if p_crack < 0.5 else 'WRONG'
    real_ok  = 'OK' if p_real > 0.5 else 'WRONG (false alarm)'
    
    all_ok = 'YES' if (clean_ok == 'OK' and crack_ok == 'OK' and real_ok == 'OK') else 'NO'
    
    print("%-14s | clean=%.3f %-6s | crack=%.3f %-6s | real=%.3f %-18s | %s" % (
        sname, p_clean, clean_ok, p_crack, crack_ok, p_real, real_ok, all_ok))
