"""
Deep diagnostic: understand WHY a clean-looking concrete image triggers CRACK prediction.
The key insight is the raw logit for our synthetic crack image was 30.59 (huge).
A real concrete image probably also gets a high logit.
Let's figure out what features the model is responding to.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

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
    
    def get_features(self, x):
        return self.features(x)

sd = torch.load('model/best_crack_model.pt', map_location='cpu', weights_only=False)
model = CrackCNN()
model.load_state_dict(sd)
model.eval()

# The image in the screenshot was a lightly-textured concrete surface
# Create several realistic versions to test

tf_no_norm = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
tf_imagenet = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

print("=== REALISTIC CONCRETE TEXTURE TESTS ===")
print()

# Key question: at what pixel VARIANCE does the model flip to CRACK?
print("Testing how TEXTURE/CONTRAST level affects prediction:")
with torch.no_grad():
    base = 180  # typical concrete brightness
    for std_pct in [0, 1, 2, 5, 10, 15, 20, 30]:
        np.random.seed(42)
        arr = np.random.normal(base, std_pct * 2.55, (64, 64, 3)).clip(0,255).astype(np.uint8)
        img = Image.fromarray(arr)
        t = tf_no_norm(img).unsqueeze(0)
        raw = model(t).item()
        prob = torch.sigmoid(torch.tensor(raw)).item()
        verdict = "CRACK" if prob > 0.5 else "NO CRACK"
        print("  std=%-3d pixels: raw=%7.2f  prob=%.4f  -> %s" % (
            int(std_pct * 2.55), raw, prob, verdict))

print()
print("=== WHAT DOES THE MODEL ACTUALLY MEASURE? ===")
print("Feature map activation magnitudes vs image std:")
with torch.no_grad():
    for std_pct in [0, 5, 10, 20, 30]:
        np.random.seed(42)
        arr = np.random.normal(180, std_pct * 2.55, (64,64,3)).clip(0,255).astype(np.uint8)
        img = Image.fromarray(arr)
        t = tf_no_norm(img).unsqueeze(0)
        feats = model.get_features(t)
        feat_mean = feats.mean().item()
        feat_max = feats.max().item()
        raw = model(t).item()
        print("  std=%3d  feat_mean=%.3f  feat_max=%.3f  raw_logit=%.2f" % (
            int(std_pct*2.55), feat_mean, feat_max, raw))

print()
print("=== TESTING: Does training data use different IMG_SIZE? ===")
print("Testing 128x128 and 224x224 input sizes:")
with torch.no_grad():
    # Check if 4096 could come from different sizes
    # 64x64 with 4 maxpool: 4x4x256 = 4096 - YES
    # 32x32 with 3 maxpool: 4x4x256 = 4096 - YES (32->16->8->4)  
    # Let's test with 32x32 input but forward through same model
    for sz in [32, 48, 56, 64, 96, 128]:
        # For non-64 sizes, the flatten will give wrong size, so we test spatial output
        test_t = torch.zeros(1, 3, sz, sz)
        try:
            feats = model.get_features(test_t)
            flat_size = feats.view(1,-1).shape[1]
            print("  Input %dx%d -> features flatten = %d (need 4096)" % (sz, sz, flat_size))
        except Exception as e:
            print("  Input %dx%d -> ERROR: %s" % (sz, sz, e))
