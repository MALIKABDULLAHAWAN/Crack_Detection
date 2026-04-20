import urllib.request, json, base64, io, os
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

img = Image.new('RGB', (64,64), color=(180,180,180))
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

payload = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": "What color is this image? Reply in one word."}
        ]}
    ],
    "temperature": 0.1, "max_tokens": 50
}

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": "Bearer " + GROQ_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    print("SUCCESS! Response:", data["choices"][0]["message"]["content"])
    print("Model used:", data.get("model"))
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print("HTTP ERROR %d: %s" % (e.code, body[:500]))
except Exception as e:
    print("ERROR:", type(e).__name__, e)
