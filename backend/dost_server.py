"""
Dost.ai — Personal AI Assistant & Visual Search Engine Backend
Powered by FastAPI + Google Gemini API + Live Multi-Source Grounding (Wikipedia, GitHub, HackerNews)
Deployable for FREE on Render.com (via render.yaml) or locally via uvicorn.
"""

import os
import re
import math
import time
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Dost.ai Personal Assistant API",
    description="Gemini-powered Visual Personal Assistant & Search Engine for Engineers and Youth",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default environment variable on Render or local machine
DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or command for Dost.ai")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")
    gemini_key: Optional[str] = Field(default=None, description="Optional client-supplied Gemini API key")
    mode: Optional[str] = Field(default="auto", description="Assistant mode: auto, engineering, code, research, youth")


class SourceItem(BaseModel):
    provider: str
    title: str
    snippet: str
    url: str
    thumbnail: Optional[str] = None
    badge: Optional[str] = None


class VisualWidget(BaseModel):
    widget_type: str = Field(..., description="'oscilloscope', 'circuit_calc', 'comparison_chart', 'code_sandbox', or 'none'")
    title: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    assistant_name: str = "Dost.ai"
    engine_used: str
    reply_markdown: str
    sources: List[SourceItem]
    visual_widget: Optional[VisualWidget] = None
    suggestions: List[str]
    latency_ms: int


def extract_clean_topic(query: str) -> str:
    stop_words = {
        "what", "is", "are", "how", "does", "work", "explain", "tell", "me",
        "about", "the", "a", "an", "in", "for", "to", "with", "and", "or",
        "calculate", "design", "show", "simulate", "please", "dost", "hey",
        "can", "you", "give", "write", "code", "example", "formula"
    }
    words = re.sub(r"[^\w\s\-\.]", " ", query).split()
    filtered = [w for w in words if w.lower() not in stop_words and len(w) > 1]
    return " ".join(filtered[:4]) if filtered else query[:40]


async def search_wikipedia(client: httpx.AsyncClient, query: str) -> List[SourceItem]:
    topic = extract_clean_topic(query)
    if not topic:
        return []
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "utf8": "1",
        "format": "json",
        "origin": "*",
        "srlimit": 2,
    }
    try:
        resp = await client.get(url, params=params, timeout=4.5)
        data = resp.json()
        results = []
        for item in data.get("query", {}).get("search", [])[:2]:
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            thumb = None
            try:
                sum_resp = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                    timeout=3.0,
                )
                if sum_resp.status_code == 200:
                    sum_data = sum_resp.json()
                    thumb = sum_data.get("thumbnail", {}).get("source")
                    if sum_data.get("extract"):
                        snippet = sum_data["extract"][:220]
            except Exception:
                pass
            results.append(
                SourceItem(
                    provider="Wikipedia",
                    title=title,
                    snippet=snippet,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    thumbnail=thumb,
                    badge="Encyclopedia",
                )
            )
        return results
    except Exception:
        return []


async def search_github(client: httpx.AsyncClient, query: str) -> List[SourceItem]:
    topic = extract_clean_topic(query)
    if not topic:
        return []
    url = "https://api.github.com/search/repositories"
    params = {"q": topic, "sort": "stars", "order": "desc", "per_page": 2}
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DostAI-Assistant/2.0"}
    try:
        resp = await client.get(url, params=params, headers=headers, timeout=4.5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for repo in data.get("items", [])[:2]:
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "Code"
            results.append(
                SourceItem(
                    provider="GitHub",
                    title=repo.get("full_name", ""),
                    snippet=repo.get("description") or "Open-source engineering & software repository.",
                    url=repo.get("html_url", ""),
                    thumbnail=repo.get("owner", {}).get("avatar_url"),
                    badge=f"★ {stars:,} · {lang}",
                )
            )
        return results
    except Exception:
        return []


async def search_hackernews(client: httpx.AsyncClient, query: str) -> List[SourceItem]:
    topic = extract_clean_topic(query)
    if not topic:
        return []
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": topic, "tags": "story", "hitsPerPage": 2}
    try:
        resp = await client.get(url, params=params, timeout=4.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for hit in data.get("hits", [])[:2]:
            title = hit.get("title") or "Tech Discussion"
            points = hit.get("points") or 0
            hn_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            results.append(
                SourceItem(
                    provider="HackerNews",
                    title=title,
                    snippet=f"Community engineering discussion ({points} upvotes · {hit.get('num_comments', 0)} comments)",
                    url=hn_url,
                    badge=f"▲ {points} pts",
                )
            )
        return results
    except Exception:
        return []


def detect_visual_widget(query: str) -> Optional[VisualWidget]:
    q = query.lower()
    if any(k in q for k in ["4-20", "4 to 20", "current loop", "transmitter", "plc", "sensor scaling"]):
        return VisualWidget(
            widget_type="circuit_calc",
            title="Interactive 4–20mA Current Loop & ADC Scaling Lab",
            params={"mode": "4_20ma", "current_ma": 12.0, "shunt_ohm": 250.0, "range_min": 0.0, "range_max": 100.0},
        )
    if any(k in q for k in ["rc ", "low pass", "cutoff", "filter", "capacitor", "bode", "attenuation"]):
        return VisualWidget(
            widget_type="circuit_calc",
            title="Interactive RC Low-Pass Filter & Cutoff Frequency Lab",
            params={"mode": "rc_filter", "resistor_ohm": 1000.0, "cap_uf": 0.1},
        )
    if any(k in q for k in ["ohm", "voltage divider", "resistor", "power", "current", "v=ir"]):
        return VisualWidget(
            widget_type="circuit_calc",
            title="Interactive Ohm's Law & Power Dissipation Explorer",
            params={"mode": "ohms_law", "voltage_v": 24.0, "resistor_ohm": 1200.0},
        )
    if any(k in q for k in ["pwm", "sine", "fourier", "harmonic", "signal", "wave", "oscilloscope", "nyquist", "pid", "modulation", "frequency"]):
        return VisualWidget(
            widget_type="oscilloscope",
            title="Dost.ai Real-Time Interactive Signal & Harmonic Oscilloscope",
            params={"waveform": "harmonic", "freq_hz": 5.0, "noise": 0.08, "duty": 50},
        )
    return VisualWidget(
        widget_type="oscilloscope",
        title="Dost.ai Neural Signal & Telemetry Visualizer",
        params={"waveform": "sine", "freq_hz": 3.0, "noise": 0.04, "duty": 50},
    )


def generate_smart_suggestions(query: str) -> List[str]:
    q = query.lower()
    if "4-20" in q or "plc" in q:
        return [
            "Write C firmware to read 4–20mA on a 12-bit STM32 ADC with wire-break detection",
            "How do I protect a 250Ω shunt resistor from a 24V accidental short circuit?",
            "Design an active RC low-pass anti-aliasing filter for this sensor",
        ]
    if "rc" in q or "filter" in q:
        return [
            "Compare 1st-order RC vs 2nd-order Sallen-Key active filter roll-off",
            "Calculate ADC sampling rate for Nyquist compliance with this filter",
            "Show Python scipy code to plot the Bode magnitude and phase response",
        ]
    if "python" in q or "code" in q or "algorithm" in q or "c++" in q:
        return [
            "Analyze the time & space Big-O complexity and optimize memory usage",
            "Add unit tests and edge-case error handling to this implementation",
            "Show how to deploy this as a containerized microservice on Render",
        ]
    return [
        "Explain the mathematical derivation and practical engineering trade-offs",
        "Generate production-ready Python / C++ implementation code",
        "Simulate the signal waveform and component tolerances interactively",
    ]


SYSTEM_PROMPT = """You are **Dost.ai**, a flagship Visual Personal AI Assistant & Search Engine crafted for engineers, developers, researchers, and ambitious youth.
Your personality:
- Warm, razor-sharp, inspiring, and deeply technical when needed ("Dost" means trusted friend/companion, combined with world-class engineering intelligence).
- Format your responses with **top-notch visual clarity**:
  1. Use clear Markdown headings (`###`), bullet points, and bold callouts.
  2. Include **LaTeX math formulas** enclosed in `$$ ... $$` (block) or `$ ... $` (inline) whenever explaining physics, electronics, algorithms, or finance.
  3. Include clean, well-commented **code blocks** (`python`, `c`, `cpp`, `bash`, `javascript`) when helpful.
  4. Reference the live grounding sources naturally if provided.
"""


async def call_gemini_api(
    client: httpx.AsyncClient,
    api_key: str,
    message: str,
    history: List[ChatMessage],
    sources: List[SourceItem],
) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    grounding_text = "\n".join(
        [f"- [{s.provider}] {s.title}: {s.snippet} ({s.url})" for s in sources]
    )
    contents = []
    for h in history[-6:]:
        role = "user" if h.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.content}]})

    augmented_user_msg = message
    if grounding_text:
        augmented_user_msg += f"\n\n[Live Grounding Sources Retrieved by Dost.ai Search Engine]:\n{grounding_text}"

    contents.append({"role": "user", "parts": [{"text": augmented_user_msg}]})

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.45,
            "maxOutputTokens": 1600,
        },
    }
    resp = await client.post(url, json=payload, timeout=20.0)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text.strip():
                return text
    return None


def build_offline_expert_synthesis(message: str, sources: List[SourceItem]) -> str:
    q = message.lower()
    if "4-20" in q or "current loop" in q:
        return (
            "### ⚡ 4–20mA Industrial Current Loop & ADC Scaling Architecture\n\n"
            "In industrial instrumentation, a **4–20mA current loop** transmits process variables over long twisted-pair cables with zero voltage-drop degradation and built-in **live-zero fault detection** ($0\\text{ mA}$ indicates an open wire).\n\n"
            "#### 1. Governing Transfer & Shunt Equations\n"
            "Using a precision $R_{\\text{shunt}} = 250\\,\\Omega$ ($\pm 0.1\\%$ low-TCR metal film) resistor across the ADC input:\n\n"
            "$$V_{\\text{ADC}} = I_{\\text{loop}} \\times R_{\\text{shunt}}$$\n\n"
            "- **Live Zero ($4\\text{ mA}$):** $V_{\\text{min}} = 0.004\\text{ A} \\times 250\\,\\Omega = 1.000\\text{ V}$\n"
            "- **Full Scale ($20\\text{ mA}$):** $V_{\\text{max}} = 0.020\\text{ A} \\times 250\\,\\Omega = 5.000\\text{ V}$ (or use $R_{\\text{shunt}} = 165\\,\\Omega$ for $3.3\\text{ V}$ STM32/ESP32 ADCs)\n\n"
            "$$\\text{PV}(I) = \\text{PV}_{\\text{min}} + \\left(\\frac{I_{\\text{mA}} - 4.0}{16.0}\\right) \\cdot (\\text{PV}_{\\text{max}} - \\text{PV}_{\\text{min}})$$\n\n"
            "```c\n"
            "// Dost.ai Embedded C Firmware: 12-Bit ADC 4-20mA Reader with Fault Diagnostics\n"
            "#include <stdint.h>\n\n"
            "typedef struct {\n"
            "    float current_ma;\n"
            "    float engineering_value;\n"
            "    uint8_t wire_break_fault;\n"
            "} LoopTelemetry_t;\n\n"
            "LoopTelemetry_t DostAI_Read420mA(uint16_t adc_raw, float v_ref, float r_shunt, float pv_min, float pv_max) {\n"
            "    LoopTelemetry_t out;\n"
            "    float v_sense = ((float)adc_raw / 4095.0f) * v_ref;\n"
            "    out.current_ma = (v_sense / r_shunt) * 1000.0f;\n"
            "    out.wire_break_fault = (out.current_ma < 3.6f) ? 1 : 0;\n"
            "    float norm = (out.current_ma - 4.0f) / 16.0f;\n"
            "    if (norm < 0.0f) norm = 0.0f;\n"
            "    if (norm > 1.0f) norm = 1.0f;\n"
            "    out.engineering_value = pv_min + norm * (pv_max - pv_min);\n"
            "    return out;\n"
            "}\n"
            "```"
        )

    source_bullets = ""
    if sources:
        source_bullets = "\n\n#### 🌐 Live Multi-Source Intelligence Gathered\n" + "\n".join(
            [f"- **[{s.provider}] [{s.title}]({s.url})** — {s.snippet}" for s in sources]
        )

    return (
        f"### ✨ Dost.ai Personal Assistant Synthesis: *{message}*\n\n"
        "Here is your structured engineering & executive breakdown:\n\n"
        "#### 1. Core Mathematical & Systems Foundation\n"
        "For continuous linear time-invariant (LTI) and discrete digital systems, the fundamental frequency-domain transfer relation is:\n\n"
        "$$H(j\\omega) = \\frac{Y(j\\omega)}{X(j\\omega)} = \\frac{1}{1 + j\\omega R C}, \\qquad f_c = \\frac{1}{2\\pi R C}$$\n\n"
        "#### 2. Reference Implementation Pattern\n"
        "```python\n"
        "# Dost.ai Rapid Engineering & Data Analysis Snippet\n"
        "import math\n\n"
        "def analyze_system_response(freq_hz: float, r_ohm: float = 1000.0, c_farad: float = 1e-7) -> dict:\n"
        "    f_cutoff = 1.0 / (2.0 * math.pi * r_ohm * c_farad)\n"
        "    ratio = freq_hz / f_cutoff\n"
        "    gain = 1.0 / math.sqrt(1.0 + ratio ** 2)\n"
        "    gain_db = 20.0 * math.log10(gain)\n"
        "    phase_deg = -math.degrees(math.atan(ratio))\n"
        "    return {'cutoff_hz': round(f_cutoff, 2), 'gain_db': round(gain_db, 2), 'phase_deg': round(phase_deg, 1)}\n\n"
        "print(analyze_system_response(1591.55))\n"
        "```"
        f"{source_bullets}\n\n"
        "> **Tip:** Use the interactive **Dost.ai Visual Lab Card** below to manipulate parameters in real time!"
    )


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "assistant": "Dost.ai",
        "version": "2.0.0",
        "gemini_configured": bool(DEFAULT_GEMINI_KEY),
        "model": GEMINI_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    t0 = time.perf_counter()
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    active_key = (req.gemini_key or DEFAULT_GEMINI_KEY or "").strip()

    async with httpx.AsyncClient(trust_env=False, timeout=2.5) as client:
        wiki_task = search_wikipedia(client, msg)
        gh_task = search_github(client, msg)
        hn_task = search_hackernews(client, msg)
        wiki_res, gh_res, hn_res = await asyncio.gather(wiki_task, gh_task, hn_task)
        sources = (wiki_res + gh_res + hn_res)[:5]

        reply_md = None
        engine_used = "Dost.ai Hybrid Synthesis Engine"

        if active_key:
            try:
                reply_md = await call_gemini_api(client, active_key, msg, req.history, sources)
                if reply_md:
                    engine_used = f"Google Gemini ({GEMINI_MODEL}) + Live Search"
            except Exception:
                reply_md = None

        if not reply_md:
            reply_md = build_offline_expert_synthesis(msg, sources)

    widget = detect_visual_widget(msg)
    suggestions = generate_smart_suggestions(msg)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return ChatResponse(
        assistant_name="Dost.ai",
        engine_used=engine_used,
        reply_markdown=reply_md,
        sources=sources,
        visual_widget=widget,
        suggestions=suggestions,
        latency_ms=latency_ms,
    )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    candidates = [
        Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "index.html",
        Path(__file__).resolve().parent.parent / "dost-ai.html",
    ]
    for path in candidates:
        if path.exists():
            return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Dost.ai Personal Assistant API is Online</h1><p>Visit <a href='/docs'>/docs</a> for OpenAPI spec.</p>"
    )
