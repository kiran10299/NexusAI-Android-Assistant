# ✨ Dost.ai — Personal AI Assistant & Visual Search Engine for Engineers & Youth

[![Live Web & PWA](https://img.shields.io/badge/Live_App-Dost.ai_Web_%26_PWA-38bdf8?style=for-the-badge)](https://kiran10299.github.io/dost-ai.html)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_%2B_Google_Gemini_2.0-818cf8?style=for-the-badge)](./backend/dost_server.py)
[![Free Hosting](https://img.shields.io/badge/Host_Free-Render.com_Blueprint-34d399?style=for-the-badge)](./render.yaml)

**Dost.ai** is a top-notch, ultra-aesthetic **Visual AI Personal Assistant & Multi-Source Search Engine** crafted for engineers, developers, researchers, and youth. It combines a **3D Volumetric Fluid & Particle Neural Core Orb**, **Google Gemini 2.0 Flash** intelligence, **real-time multi-source search grounding** (Wikipedia, GitHub, HackerNews), **rendered LaTeX math & syntax-highlighted code**, and **interactive inline engineering labs** (60 FPS Oscilloscope, 4–20mA Current Loop Simulator, and RC Low-Pass Filter Bode Explorer).

---

## 🚀 Try Dost.ai Live Right Now

- **Live Web & Installable Mobile PWA:** [**https://kiran10299.github.io/dost-ai.html**](https://kiran10299.github.io/dost-ai.html)

---

## ☁️ How to Host the Python Gemini Backend for FREE on Render.com (2 Minutes)

You don't need to pay for a server! This repository includes a ready-to-run **FastAPI Python Backend** (`backend/dost_server.py`) and a 1-click **Render Blueprint** (`render.yaml`).

1. **Get a Free Google Gemini API Key:**
   - Visit [**Google AI Studio (aistudio.google.com)**](https://aistudio.google.com/app/apikey) and click **Create API Key** (100% free tier).
   - *(Tip: You can also paste this key directly inside the **⚡ Gemini / Render Setup** modal in the web app without deploying anything!)*
2. **Deploy on Render.com (Free Tier):**
   - Sign in to [**render.com**](https://render.com/) with your GitHub account.
   - Click **New +** → **Blueprint** and select this repository (`kiran10299/NexusAI-Android-Assistant`).
   - Render will automatically detect `render.yaml` and prompt you to paste your `GEMINI_API_KEY`.
   - Click **Apply** — Render will build and host your personal **Dost.ai** FastAPI server + frontend at `https://dost-ai-personal-assistant.onrender.com` for free!

---

## ✨ Key Architectural Highlights

1. **3D Volumetric Neural Core Orb (`Dost.ai`):**
   - High-DPI HTML5 Canvas rendering a Fibonacci 3D particle sphere and multi-layered harmonic plasma rings that dynamically morph across `IDLE`, `LISTENING`, `SEARCHING`, and `SPEAKING` states.
2. **Rich Rendered Visual Answers:**
   - **Perplexity-Style Live Citation Carousel:** Parallel search across Wikipedia REST API, GitHub Repositories API, and HackerNews Algolia API.
   - **KaTeX Mathematical Typesetting & Highlight.js Code Blocks:** Crisp block/inline LaTeX equations (`$$...$$`) and syntax-highlighted C/Python/C++ code with 1-click **Copy Code** buttons.
   - **Interactive Inline Engineering Labs:**
     - **4–20mA Current Loop & 12-Bit ADC Simulator:** Live sliders for loop current ($0\text{–}24\text{ mA}$) and shunt resistance ($100\text{–}500\,\Omega$) with wire-break fault diagnostics.
     - **RC Low-Pass Filter & Bode Cutoff Lab:** Real-time $-3\text{ dB}$ cutoff frequency ($f_c$), time constant ($\tau$), and Nyquist sampling rate calculator.
     - **60 FPS Harmonic Oscilloscope:** Real-time waveform synthesizer with Fundamental Frequency, Harmonic Overtones, and Noise Floor sliders inside the chat response.
3. **Daily Youth & Engineer Focus Companion Drawer:**
   - Built-in **25:00 Deep-Work Pomodoro Timer**, **Instant Voltage Ratio to dB Converter**, and **Daily Engineering Growth Checklist**.

---

## 🛠️ Local Python Backend Quickstart

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
uvicorn backend.dost_server:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser!
