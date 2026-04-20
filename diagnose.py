import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

def label(p):
    if p > 0.5:
        return "CRACK (%s%%)" % round(p*100, 1)
    else:
        return "NO CRACK (%s%%)" % round((1-p)*100, 1)

print("=== MODEL BIAS TEST ===")
with torch.no_grad():
    tests = [
        ("Black image (all 0)",       torch.zeros(1,3,64,64)),
        ("White /255 (all 1.0)",       torch.ones(1,3,64,64)),
        ("Mid gray 0.5",               torch.full((1,3,64,64), 0.5)),
        ("Concrete ~200/255",          torch.full((1,3,64,64), 200/255)),
        ("Dark ~50/255",               torch.full((1,3,64,64), 50/255)),
    ]
    for name, t in tests:
        p = torch.sigmoid(model(t)).item()
        raw = model(t).item()
        print("  %-30s  raw=%6.2f  prob=%.4f  -> %s" % (name, raw, p, label(p)))

print()
print("Classifier bias:", sd['classifier.6.bias'].item())
print("Classifier weight norm:", sd['classifier.6.weight'].norm().item())
print()

# Check what normalization would produce sensible output
# Typical crack image dataset: pixel values ~100-200, gray concrete
print("=== NORMALIZATION SENSITIVITY ===")
with torch.no_grad():
    for pixel_val in [100, 150, 200]:
        raw_val = pixel_val / 255.0
        imagenet_val = (raw_val - 0.485) / 0.229
        std05_val = (raw_val - 0.5) / 0.5
        
        t_raw = torch.full((1,3,64,64), raw_val)
        t_imgnet = torch.full((1,3,64,64), imagenet_val)
        t_05 = torch.full((1,3,64,64), std05_val)
        
        p_raw = torch.sigmoid(model(t_raw)).item()
        p_imgnet = torch.sigmoid(model(t_imgnet)).item()
        p_05 = torch.sigmoid(model(t_05)).item()
        
        print("  pixel=%d: no-norm=%.3f  imagenet=%.3f  std0.5=%.3f" % (
            pixel_val, p_raw, p_imgnet, p_05))
