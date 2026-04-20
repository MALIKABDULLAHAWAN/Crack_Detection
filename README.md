# Crack Detection System — FYP Web App

A dual-engine crack detection system using a custom CNN model + Groq AI (Llama 4 Vision).

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MALIKABDULLAHAWAN/Crack_Detection.git
cd Crack_Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```
Then edit `.env` and set your `GROQ_API_KEY`.

### 4. Add your model file
Place `best_crack_model.pt` inside the `model/` folder.

### 5. Run the app
```bash
python app.py
```

Open your browser at: `http://127.0.0.1:5000`

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (get one at console.groq.com) |
| `GROQ_MODEL` | Groq model to use (default: llama-4-scout-17b) |
| `GROQ_URL` | Groq API endpoint |
| `FLASK_PORT` | Port to run Flask on (default: 5000) |
| `FLASK_HOST` | Host to bind to (default: 0.0.0.0) |

> **Never commit your `.env` file.** It is listed in `.gitignore`.

---

## Project Structure
```
crack_app/
├── app.py                  ← Flask app (CNN + Groq inference)
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .env                    ← Your local secrets (not committed)
├── .gitignore
├── run.sh                  ← Quick start script (Linux/Mac)
├── model/
│   └── best_crack_model.pt ← Trained CNN weights (not committed)
├── static/
│   ├── sample_crack.jpg
│   └── sample_nocrack.jpg
└── templates/
    └── index.html
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: dotenv` | Run `pip install python-dotenv` |
| `GROQ_API_KEY not set` | Check your `.env` file |
| Page not loading | Make sure `app.py` is still running |

---
FYP — CNN-based Crack Detection for Structural Health Monitoring
