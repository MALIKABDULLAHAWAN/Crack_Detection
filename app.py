"""
Crack Detection System
======================
Dual-engine approach:
  1. CNN model  - your trained PyTorch model
  2. Groq AI    - Llama 4 vision model for intelligent image analysis

Both engines analyze the image and a combined verdict is produced.
"""

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import base64
import json
import urllib.request
import urllib.error
import numpy as np
import logging
import uuid
import time


# -------------------------------------------------------
# CONFIGURE AUDIT LOGGING
# -------------------------------------------------------
logging.basicConfig(
    filename='inference_audit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CrackApp")
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# -------------------------------------------------------
# GROQ CONFIG
# -------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_URL     = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")


# -------------------------------------------------------
# EXACT MODEL ARCHITECTURE (reconstructed from state dict)
# Custom 4-block CNN trained on 64x64 crack images
# -------------------------------------------------------

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


# -------------------------------------------------------
# LOAD CNN MODEL
# -------------------------------------------------------

IMG_SIZE   = 64
THRESHOLD  = 0.2   # Calibrated on SDNET2018: 75% accuracy at this threshold

logger.info("="*60)
logger.info("SYSTEM BOOT: Initializing Dual-Engine Crack Detection System")
print("=" * 60)
print("  Crack Detection System  [CNN + Groq AI]  Starting...")
print("=" * 60)

MODEL_PATH = os.path.join("model", "best_crack_model.pt")

try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cnn_model = CrackCNN()
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()
    logger.info("CNN Model loaded successfully from %s", MODEL_PATH)
    print("  [OK] CNN model loaded from %s" % MODEL_PATH)
    CNN_LOADED = True
except Exception as e:
    logger.error("Failed to load CNN model: %s", str(e))
    print("  [FAIL] CNN model: %s" % e)
    cnn_model = None
    CNN_LOADED = False


# -------------------------------------------------------
# CNN PREPROCESSING & INFERENCE
# -------------------------------------------------------

# ImageNet normalization — confirmed best on SDNET2018 (74.7% accuracy)
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def cnn_predict(image_bytes):
    """Run CNN model. Returns (label, confidence, raw_prob)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(tensor)
        prob = torch.sigmoid(output).item()

    if prob > THRESHOLD:
        return "CRACK DETECTED", round(prob * 100, 1), round(prob, 4)
    else:
        return "NO CRACK", round((1 - prob) * 100, 1), round(prob, 4)


# -------------------------------------------------------
# GROQ VISION INFERENCE  (pure urllib - no extra packages)
# -------------------------------------------------------

def groq_predict(image_bytes):
    """
    Send image to Groq Llama 4 vision model.
    Returns (label, confidence, explanation, severity).
    """
    # Encode image as base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Detect MIME type
    mime = "image/jpeg"
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif image_bytes[:4] == b'RIFF':
        mime = "image/webp"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert structural engineer specializing in concrete crack detection "
                    "and Structural Health Monitoring (SHM). Analyze images of concrete surfaces "
                    "and determine if structural cracks are present. "
                    "Always respond ONLY with valid JSON, no markdown, no extra text."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s" % (mime, b64)
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this concrete/structural surface image for cracks. "
                            "Respond with ONLY this JSON format (no extra text):\n"
                            "{\n"
                            '  "crack_detected": true or false,\n'
                            '  "confidence": 0-100 (integer),\n'
                            '  "severity": "None" or "Hairline" or "Minor" or "Moderate" or "Severe",\n'
                            '  "explanation": "one clear sentence explaining your finding",\n'
                            '  "insight": "A detailed 2-3 sentence structural insight analyzing the image context, surface texture, and condition.",\n'
                            '  "recommendation": "one action sentence for the engineer"\n'
                            "}"
                        )
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"}
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Authorization": "Bearer %s" % GROQ_API_KEY,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        crack = parsed.get("crack_detected", False)
        conf  = int(parsed.get("confidence", 75))
        sev   = parsed.get("severity", "None")
        expl  = parsed.get("explanation", "")
        ins   = parsed.get("insight", "")
        rec   = parsed.get("recommendation", "")

        label = "CRACK DETECTED" if crack else "NO CRACK"
        return label, conf, expl, ins, sev, rec, None

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
        logger.error("Groq API HTTP Error: %s", err_body[:200])
        return None, 0, "", "", "Unknown", "", "Groq API error: %s" % err_body[:200]
    except Exception as e:
        logger.error("Groq API Exception: %s", str(e))
        return None, 0, "", "", "Unknown", "", "Groq error: %s" % str(e)[:200]


# -------------------------------------------------------
# COMBINED VERDICT LOGIC
# -------------------------------------------------------

def combined_verdict(cnn_label, cnn_conf, groq_label, groq_conf):
    """
    Combine CNN + Groq results into a final verdict.
    Groq (vision LLM) gets 70% weight, CNN gets 30%.
    """
    if groq_label is None:
        # Groq failed — use CNN only
        return cnn_label, cnn_conf, "cnn_only"

    groq_crack_prob = groq_conf / 100.0 if groq_label == "CRACK DETECTED" else (100 - groq_conf) / 100.0
    cnn_crack_prob  = cnn_conf  / 100.0 if cnn_label  == "CRACK DETECTED" else (100 - cnn_conf)  / 100.0

    # Weighted combination: Groq 70%, CNN 30%
    combined_prob = 0.70 * groq_crack_prob + 0.30 * cnn_crack_prob

    if combined_prob > 0.5:
        return "CRACK DETECTED", round(combined_prob * 100, 1), "combined"
    else:
        return "NO CRACK", round((1 - combined_prob) * 100, 1), "combined"


# -------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "running", "cnn_loaded": CNN_LOADED})


@app.route("/predict", methods=["POST"])
def predict():
    tx_id = "TX-" + uuid.uuid4().hex[:6].upper()
    start_time = time.time()

    if "file" not in request.files:
        logger.warning("%s | Request rejected: No image received", tx_id)
        return jsonify({"error": "No image received"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.warning("%s | Request rejected: No file selected", tx_id)
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    file_size_kb = len(image_bytes) / 1024
    logger.info("%s | Request accepted: %s (%.1f KB)", tx_id, file.filename, file_size_kb)

    response = {"tx_id": tx_id}

    # --- CNN inference ---
    if CNN_LOADED:
        try:
            c_label, c_conf, c_raw = cnn_predict(image_bytes)
            response["cnn"] = {
                "label": c_label,
                "confidence": c_conf,
                "raw_probability": c_raw
            }
            logger.info("%s | CNN Verdict: %s (Conf: %.1f%%)", tx_id, c_label, c_conf)
        except Exception as e:
            logger.error("%s | CNN Execution Failed: %s", tx_id, str(e))
            response["cnn"] = {"error": str(e)}
            c_label, c_conf = "NO CRACK", 50.0
    else:
        logger.error("%s | CNN model not loaded", tx_id)
        response["cnn"] = {"error": "CNN model not loaded"}
        c_label, c_conf = "NO CRACK", 50.0

    # --- Groq vision inference ---
    g_label, g_conf, g_expl, g_ins, g_sev, g_rec, g_err = groq_predict(image_bytes)
    if g_err:
        logger.error("%s | Groq Execution Failed: %s", tx_id, g_err)
        response["groq"] = {"error": g_err}
    else:
        logger.info("%s | Groq Verdict: %s (Conf: %.1f%%, Sev: %s)", tx_id, g_label, g_conf, g_sev)
        response["groq"] = {
            "label": g_label,
            "confidence": g_conf,
            "severity": g_sev,
            "explanation": g_expl,
            "insight": g_ins,
            "recommendation": g_rec
        }

    # --- Combined verdict ---
    final_label, final_conf, source = combined_verdict(
        c_label, c_conf, g_label, g_conf
    )
    response["final"] = {
        "label": final_label,
        "confidence": final_conf,
        "source": source
    }

    latency = round((time.time() - start_time) * 1000)
    response["latency_ms"] = latency

    logger.info("%s | Final Verdict: %s (Conf: %.1f%%) | Latency: %d ms", tx_id, final_label, final_conf, latency)
    logger.info("-" * 60)

    return jsonify(response)


# -------------------------------------------------------
# START SERVER
# -------------------------------------------------------

if __name__ == "__main__":
    print("\n  Open your browser at:  http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
