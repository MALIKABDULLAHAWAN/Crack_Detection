"""End-to-end test using requests-style multipart"""
import urllib.request, json, io, uuid
from PIL import Image
import numpy as np

# Create a synthetic cracked concrete image
arr = np.random.normal(180, 12, (200, 200, 3)).clip(0,255).astype('uint8')
for i in range(30, 170):
    for d in [-1, 0, 1]:
        if 0 <= i+d < 200:
            arr[i, i+d] = [10, 10, 10]
img = Image.fromarray(arr)
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=90)
img_bytes = buf.getvalue()

# Proper multipart encoding
bnd = ('----WebKitFormBoundary' + uuid.uuid4().hex[:16]).encode()
body  = b'--' + bnd + b'\r\n'
body += b'Content-Disposition: form-data; name="file"; filename="crack_test.jpg"\r\n'
body += b'Content-Type: image/jpeg\r\n'
body += b'\r\n'
body += img_bytes
body += b'\r\n'
body += b'--' + bnd + b'--\r\n'

req = urllib.request.Request(
    'http://127.0.0.1:5000/predict',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + bnd.decode()},
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=50) as r:
        data = json.loads(r.read())
    print("=== RESULTS ===")
    print("FINAL:  ", json.dumps(data.get('final'), indent=2))
    print("CNN:    ", json.dumps(data.get('cnn'), indent=2))
    g = data.get('groq', {})
    if 'error' in g:
        print("GROQ ERROR:", g['error'][:300])
    else:
        print("GROQ label:", g.get('label'), "| conf:", g.get('confidence'), "| severity:", g.get('severity'))
        print("GROQ expl:", g.get('explanation'))
        print("GROQ rec:", g.get('recommendation'))
except urllib.error.HTTPError as e:
    print("HTTP Error %d:" % e.code, e.read().decode()[:400])
except Exception as e:
    print("Error:", type(e).__name__, e)
