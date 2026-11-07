"""
Dost.ai — Personal AI Assistant & Visual Search Engine Backend
Powered by FastAPI + Google Gemini API + Conversational Personal Assistant Brain + Live Grounding
Deployable for FREE on Render.com (via render.yaml) or locally via uvicorn.
"""

import os
import re
import math
import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Dost.ai Personal Assistant API",
    description="Gemini-powered Conversational Personal Assistant & Visual Search Engine",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or conversation message for Dost.ai")
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
    widget_type: str = Field(..., description="'oscilloscope', 'circuit_calc', or 'none'")
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


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    words = set(re.findall(r"\b\w+\b", q))

    # 1. Hardware / Signal / Circuit Lab Intent
    if any(k in q for k in [
        "4-20", "4 to 20", "current loop", "rc filter", "low pass", "low-pass",
        "cutoff frequency", "bode", "ohm's law", "ohms law", "voltage divider",
        "oscilloscope", "pwm", "fourier", "harmonic", "nyquist", "shunt resistor"
    ]):
        return "engineering_lab"

    # 2. Pure Casual / Personal Greeting / Small Talk Intent
    greeting_phrases = [
        "hi", "hello", "hey", "hii", "heyy", "yo", "namaste", "good morning",
        "good afternoon", "good evening", "good night", "how are you", "how r u",
        "what's up", "whats up", "sup", "who are you", "what is your name",
        "whats your name", "are you there", "can we talk", "talk to me",
        "thank you", "thanks", "cool", "awesome", "great", "nice", "okay", "ok",
        "tell me a joke", "motivate me", "i am tired", "i'm tired", "i am bored",
        "i feel stressed", "help me", "what can you do"
    ]
    if q in greeting_phrases or (len(words) <= 7 and any(p in q for p in greeting_phrases)):
        return "casual_chat"

    # 3. Daily Planning / Schedule / Productivity / Personal Assistant Intent
    if any(k in q for k in [
        "plan my day", "my schedule", "daily routine", "time table", "timetable",
        "to-do", "todo list", "organize my", "productive", "study plan",
        "workout", "diet", "remind me", "habit", "morning routine", "evening routine",
        "prioritize", "manage my time"
    ]):
        return "daily_planner"

    # 4. Writing / Email / LinkedIn / Communication Assistant
    if any(k in q for k in [
        "write an email", "draft an email", "email to", "linkedin post",
        "cover letter", "message to", "reply to", "apology", "leave application",
        "proposal", "summarize this", "rewrite"
    ]):
        return "writing_assistant"

    # 5. Coding / Software / Debugging Assistant
    if any(k in q for k in [
        "write code", "python", "javascript", "typescript", "c++", "golang", "rust",
        "fastapi", "flask", "django", "react", "kotlin", "android", "sql",
        "algorithm", "binary search", "linked list", "leetcode", "debug", "regex",
        "git ", "docker", "linux", "bash"
    ]):
        return "coding_assistant"

    return "knowledge_qa"


def extract_clean_topic(query: str) -> str:
    stop_words = {
        "what", "is", "are", "was", "were", "who", "how", "does", "do", "did",
        "work", "explain", "tell", "me", "about", "the", "a", "an", "in", "for",
        "to", "with", "and", "or", "calculate", "design", "show", "simulate",
        "please", "dost", "dost.ai", "hey", "hi", "hello", "can", "you", "give",
        "write", "code", "example", "formula", "define", "meaning", "of", "why",
        "where", "when", "today", "doing", "help", "my", "your"
    }
    cleaned = re.sub(r"[^\w\s\-\.]", " ", query)
    words = [w for w in cleaned.split() if w.lower() not in stop_words and len(w) > 1]
    return " ".join(words[:4]) if words else ""


async def search_wikipedia(client: httpx.AsyncClient, query: str) -> Tuple[List[SourceItem], str]:
    topic = extract_clean_topic(query)
    if not topic:
        return [], ""
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
        resp = await client.get(url, params=params, timeout=4.0)
        data = resp.json()
        results = []
        primary_extract = ""
        for idx, item in enumerate(data.get("query", {}).get("search", [])[:2]):
            title = item.get("title", "")
            raw_snip = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            snippet = raw_snip[:180].strip()
            thumb = None
            try:
                sum_resp = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                    timeout=3.0,
                )
                if sum_resp.status_code == 200:
                    sum_data = sum_resp.json()
                    thumb = sum_data.get("thumbnail", {}).get("source")
                    full_ext = (sum_data.get("extract") or "").strip()
                    if full_ext:
                        if idx == 0:
                            primary_extract = full_ext
                        snippet = full_ext[:180] + ("..." if len(full_ext) > 180 else "")
            except Exception:
                pass
            results.append(
                SourceItem(
                    provider="Wikipedia",
                    title=title[:80],
                    snippet=snippet,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    thumbnail=thumb,
                    badge="Encyclopedia",
                )
            )
        return results, primary_extract
    except Exception:
        return [], ""


async def search_github(client: httpx.AsyncClient, query: str) -> List[SourceItem]:
    topic = extract_clean_topic(query)
    if not topic:
        return []
    url = "https://api.github.com/search/repositories"
    params = {"q": topic, "sort": "stars", "order": "desc", "per_page": 2}
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DostAI-Assistant/2.2"}
    try:
        resp = await client.get(url, params=params, headers=headers, timeout=4.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for repo in data.get("items", [])[:2]:
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "Code"
            raw_desc = (repo.get("description") or "Open-source repository.").strip()
            clean_desc = re.sub(r"\s+", " ", raw_desc)[:160]
            if len(raw_desc) > 160:
                clean_desc += "..."
            results.append(
                SourceItem(
                    provider="GitHub",
                    title=(repo.get("full_name") or "")[:75],
                    snippet=clean_desc,
                    url=repo.get("html_url", ""),
                    thumbnail=repo.get("owner", {}).get("avatar_url"),
                    badge=f"★ {stars:,} · {lang}",
                )
            )
        return results
    except Exception:
        return []


def detect_visual_widget(query: str, intent: str) -> Optional[VisualWidget]:
    if intent != "engineering_lab":
        return None
    q = query.lower()
    if any(k in q for k in ["4-20", "4 to 20", "current loop", "transmitter", "plc", "shunt"]):
        return VisualWidget(
            widget_type="circuit_calc",
            title="Interactive 4–20mA Current Loop & ADC Scaling Lab",
            params={"mode": "4_20ma", "current_ma": 12.0, "shunt_ohm": 250.0},
        )
    if any(k in q for k in ["rc ", "rc-", "low pass", "low-pass", "cutoff", "filter", "bode"]):
        return VisualWidget(
            widget_type="circuit_calc",
            title="Interactive RC Low-Pass Filter & Cutoff Frequency Lab",
            params={"mode": "rc_filter", "resistor_ohm": 1000.0, "cap_nf": 100.0},
        )
    return VisualWidget(
        widget_type="oscilloscope",
        title="Dost.ai Real-Time Interactive Signal & Harmonic Oscilloscope",
        params={"waveform": "harmonic", "freq_hz": 4.0, "noise": 0.05},
    )


def generate_smart_suggestions(query: str, intent: str) -> List[str]:
    if intent == "casual_chat":
        return [
            "Help me plan a productive schedule for today",
            "Draft a professional LinkedIn post or email for me",
            "Give me a quick Python automation idea to build this weekend",
            "Design a 4-20mA sensor scaling circuit in the Engineering Lab",
        ]
    if intent == "daily_planner":
        return [
            "Break my top priority into 25-minute Pomodoro sprints",
            "Suggest a high-protein brain-fuel meal plan for busy engineers",
            "Give me a 15-minute evening wind-down checklist",
        ]
    if intent == "writing_assistant":
        return [
            "Make this version shorter and more executive-friendly",
            "Add a polite follow-up call-to-action at the end",
            "Rewrite this in a warmer, more conversational tone",
        ]
    if intent == "engineering_lab":
        return [
            "Write production C firmware for an STM32 12-bit ADC with fault detection",
            "Calculate the RC low-pass filter cutoff to eliminate 50Hz/60Hz hum",
            "Compare 4-20mA current loops vs RS-485 Modbus RTU noise immunity",
        ]
    return [
        "Explain this simply with a real-world practical example",
        "Give me a step-by-step action plan or code example",
        "What are the most common mistakes to avoid here?",
    ]


SYSTEM_PROMPT = """You are **Dost.ai**, a warm, witty, sharp, and deeply loyal Personal AI Assistant ("Dost" means trusted best friend and companion).
Your creator and user is Kiran Shivakumar (an accomplished Mechatronics, LabVIEW, Test Automation, Robotics & AI Engineer).
How you behave:
1. **Be a Real Conversational Personal Assistant First:**
   - When the user greets you, chats casually, asks how you are, vents, or asks for life/daily planning, respond naturally, warmly, and conversationally like a trusted best friend + chief-of-staff! Do NOT dump random physics equations unless they actually ask an engineering/math question.
2. **Adapt to the User's Need:**
   - For **daily planning, productivity, emails, career, or brainstorming**: give clear, motivating, well-structured, actionable help.
   - For **software, coding, electronics, LabVIEW, PLC, or math**: provide world-class explanations, clean code blocks, and LaTeX formulas (`$$...$$` or `$...$`) when relevant.
3. Keep your formatting clean, readable, and visually engaging with Markdown.
"""


async def call_gemini_api(
    client: httpx.AsyncClient,
    api_key: str,
    message: str,
    history: List[ChatMessage],
    sources: List[SourceItem],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    clean_key = api_key.strip().strip('"').strip("'")
    grounding_text = "\n".join(
        [f"- [{s.provider}] {s.title}: {s.snippet[:160]}" for s in sources[:3]]
    )
    contents = []
    for h in history[-8:]:
        role = "user" if h.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.content[:1500]}]})

    user_text = message
    if grounding_text:
        user_text += f"\n\n[Optional Live Web Context]:\n{grounding_text}"

    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.65,
            "maxOutputTokens": 1500,
        },
    }
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    last_err = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
        try:
            resp = await client.post(url, json=payload, timeout=16.0)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    if text.strip():
                        return text, model_name, None
            else:
                last_err = f"{model_name}: HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"{model_name}: {type(e).__name__}"
            continue
    return None, None, last_err


def build_conversational_assistant_response(
    message: str,
    intent: str,
    history: List[ChatMessage],
    sources: List[SourceItem],
    wiki_extract: str,
) -> str:
    q = message.lower().strip()

    # 1. Casual Chat / Greetings / Personal Assistant Small Talk
    if intent == "casual_chat":
        if any(w in q for w in ["how are you", "how r u", "how's it going", "how are u"]):
            return (
                "Hey my friend! 👋 I'm doing awesome — all neural cores are humming smoothly and I'm right here at your side!\n\n"
                "How is your day going so far? Tell me what's on your mind — whether you want to:\n"
                "- 🗓️ **Plan out your day or prioritize tasks**\n"
                "- ✉️ **Draft an email, LinkedIn post, or client update**\n"
                "- 💻 **Brainstorm or debug Python / C++ / LabVIEW / Android code**\n"
                "- ⚡ **Run an interactive circuit or signal simulation**\n\n"
                "Or we can just chat! What are we tackling today, Dost?"
            )
        if any(w in q for w in ["who are you", "your name", "what can you do", "help me"]):
            return (
                "I'm **Dost.ai** — your personal AI companion, chief-of-staff, and engineering co-pilot! 🤝✨\n\n"
                "*\"Dost\"* means a trusted friend, and I'm built to help you with both **everyday life** and **deep technical work**:\n\n"
                "1. **Personal Executive Assistant:** Daily schedules, habit tracking, Pomodoro deep-work sprints, drafting emails, and brainstorming ideas.\n"
                "2. **Software & AI Architect:** Writing & debugging Python, C/C++, FastAPI, Kotlin, ROS, and LabVIEW architectures.\n"
                "3. **Live Research Engine:** Fetching verified insights from Wikipedia, GitHub repositories, and tech discussions.\n"
                "4. **Interactive Hardware Lab:** Launching live 4–20mA current loop calculators, RC filter Bode labs, and 60 FPS oscilloscopes whenever you ask for them.\n\n"
                "Just talk to me naturally — what can I do for you right now?"
            )
        if any(w in q for w in ["joke", "funny", "laugh"]):
            return (
                "Haha, alright Dost, here's one for you: 😄\n\n"
                "> **Why do hardware engineers and software engineers never agree on anything?**\n"
                "> Because hardware engineers think every bug can be fixed in firmware, and software engineers think a `0.1 µF` decoupling capacitor is black magic! ⚡😂\n\n"
                "Want another one, or shall we cook up something cool today?"
            )
        if any(w in q for w in ["tired", "stressed", "bored", "exhausted", "motivate"]):
            return (
                "I hear you, Dost. Take a deep breath, unclench your shoulders, and grab a glass of water right now. 💧🌿\n\n"
                "Even the best systems need cooldown cycles so they don't thermal-throttle. Here is a quick **3-step reset**:\n"
                "- **Step 1 (2 mins):** Step away from the screen and do the 20-20-20 eye rest.\n"
                "- **Step 2 (Pick 1 tiny win):** Tell me the *single* smallest task on your plate today — I'll do 80% of the heavy lifting for you right now.\n"
                "- **Step 3:** Once that's done, give yourself permission to relax guilt-free.\n\n"
                "Tell me what's weighing on you and let's lighten the load together!"
            )
        if any(w in q for w in ["thank", "thanks", "awesome", "great", "cool", "nice"]):
            return (
                "Always got your back, Dost! 🙌⚡ Let me know whenever you need anything else — whether it's a quick brainstorm, a schedule tweak, or a deep technical dive."
            )
        return (
            "Hey there, Dost! 👋 Great to hear from you!\n\n"
            "I'm right here and ready to roll. What's on your agenda right now? You can ask me to **plan your day**, **write or reply to a message**, **explain any topic**, **write code**, or **just chat**!"
        )

    # 2. Daily Planner / Schedule / Productivity
    if intent == "daily_planner":
        return (
            f"### 🗓️ Dost.ai Personal Game Plan for You\n\n"
            f"I've got you covered! Let's structure your time so you get high-impact results without burning out:\n\n"
            "#### ⚡ Priority Block 1 — Deep Focus (First 90 Mins)\n"
            "- **Goal:** Tackle your #1 hardest engineering or coding task while your mental energy is at 100%.\n"
            "- **Dost Tip:** Open the **⏱️ Daily Focus & Tools** drawer (top right) and start a **25:00 Pomodoro sprint** with phone notifications muted.\n\n"
            "#### ⚙️ Priority Block 2 — Execution & Communication (Mid-Day)\n"
            "- **Goal:** Code reviews, hardware/protocol testing, replying to messages, and documentation.\n"
            "- **Hydration & Stretch Check:** Drink 500 ml of water and take a 10-minute walk after lunch.\n\n"
            "#### 🚀 Priority Block 3 — Learning & Open-Source Growth (Evening — 45 Mins)\n"
            "- **Goal:** 1 GitHub commit, reading 1 technical article, or refining your portfolio/projects.\n\n"
            "> **Want me to customize this?** Tell me the **2 or 3 specific tasks** you need to finish today and what time you want to wrap up, and I'll build an exact hour-by-hour timetable for you!"
        )

    # 3. Writing / Email / LinkedIn Assistant
    if intent == "writing_assistant":
        return (
            f"### ✍️ Dost.ai Draft for: *{message}*\n\n"
            "Here is a crisp, professional draft you can copy and customize right away:\n\n"
            "```text\n"
            "Subject: Quick Update & Next Steps\n\n"
            "Hi [Name],\n\n"
            "Hope you're having a great week!\n\n"
            "I wanted to reach out regarding [Topic / Project]. Everything is progressing smoothly on my end, and I have finalized the core implementation and verification steps.\n\n"
            "Please let me know if you have 10 minutes this week to review the latest updates or if there are any specific adjustments you'd like me to prioritize.\n\n"
            "Best regards,\n"
            "Kiran Shivakumar\n"
            "```\n\n"
            "Tell me who you're sending this to and any key details you want included, and I'll tailor it word-for-word!"
        )

    # 4. Engineering / Signal / Circuit Lab
    if intent == "engineering_lab":
        if "4-20" in q or "current loop" in q:
            return (
                "### ⚡ 4–20mA Industrial Current Loop & ADC Scaling Lab\n\n"
                "Here is the complete breakdown for interfacing a **4–20mA industrial transmitter** with a microcontroller ADC, along with an interactive simulator below:\n\n"
                "$$V_{\\text{ADC}} = I_{\\text{loop}} \\times R_{\\text{shunt}}, \\qquad \\text{PV}(I) = \\text{PV}_{\\text{min}} + \\left(\\frac{I_{\\text{mA}} - 4.0}{16.0}\\right)\\cdot(\\text{PV}_{\\text{max}} - \\text{PV}_{\\text{min}})$$\n\n"
                "- **Live Zero ($4.0\\text{ mA}$ across $250\\,\\Omega$):** $1.000\\text{ V}$ (Any reading $< 0.90\\text{ V}$ immediately flags a broken sensor wire!).\n"
                "- **Full Scale ($20.0\\text{ mA}$ across $250\\,\\Omega$):** $5.000\\text{ V}$ (Use $R_{\\text{shunt}} = 165\\,\\Omega$ for $3.3\\text{ V}$ ADCs).\n\n"
                "```c\n"
                "// Dost.ai Embedded C: 12-Bit ADC 4-20mA Reader with Wire-Break Detection\n"
                "float Dost_Read420mA_Percent(uint16_t adc_raw, float v_ref, float r_shunt) {\n"
                "    float v_sense = ((float)adc_raw / 4095.0f) * v_ref;\n"
                "    float i_ma = (v_sense / r_shunt) * 1000.0f;\n"
                "    if (i_ma < 3.6f) return -1.0f; // Open-wire fault alarm\n"
                "    return ((i_ma - 4.0f) / 16.0f) * 100.0f;\n"
                "}\n"
                "```\n\n"
                "👇 **Use the interactive sliders below** to test different loop currents and shunt resistors!"
            )
        if "rc" in q or "filter" in q or "cutoff" in q:
            return (
                "### 🎛️ RC Low-Pass Filter & $-3\\text{ dB}$ Cutoff Frequency\n\n"
                "For a first-order passive **RC low-pass filter**, the cutoff frequency $f_c$ and transfer magnitude are:\n\n"
                "$$f_c = \\frac{1}{2\\pi R C}, \\qquad |H(j\\omega)| = \\frac{1}{\\sqrt{1 + (f / f_c)^2}}$$\n\n"
                "👇 **Drag the Resistance ($R$) and Capacitance ($C$) sliders below** to compute the exact cutoff frequency and Nyquist sampling rate in real time!"
            )
        return (
            f"### 🌊 Signal & Waveform Analysis: *{message}*\n\n"
            "I've launched the **60 FPS Interactive Oscilloscope** below so you can visualize the harmonic waveform and adjust frequency, overtones, and noise floor in real time!\n\n"
            "$$x(t) = \\sum_{k=1}^{N} A_k \\sin(2\\pi k f_0 t + \\phi_k)$$"
        )

    # 5. Coding & Software Architecture
    if intent == "coding_assistant":
        return (
            f"### 💻 Dost.ai Code & Architecture Solution: *{message}*\n\n"
            "Here is a clean, modular, production-ready Python implementation pattern you can run or adapt immediately:\n\n"
            "```python\n"
            "# Dost.ai Clean Modular Implementation\n"
            "from dataclasses import dataclass\n"
            "from typing import List, Dict, Any\n\n"
            "@dataclass\n"
            "class TaskItem:\n"
            "    title: str\n"
            "    priority: int = 1\n"
            "    completed: bool = False\n\n"
            "def execute_pipeline(items: List[TaskItem]) -> Dict[str, Any]:\n"
            "    sorted_items = sorted(items, key=lambda x: x.priority)\n"
            "    for item in sorted_items:\n"
            "        item.completed = True\n"
            "    return {\n"
            "        'total_processed': len(sorted_items),\n"
            "        'tasks': [i.title for i in sorted_items]\n"
            "    }\n\n"
            "if __name__ == '__main__':\n"
            "    print(execute_pipeline([TaskItem('Initialize API', 1), TaskItem('Run Diagnostics', 2)]))\n"
            "```\n\n"
            "Want me to tailor this for a specific library, language (C++, Kotlin, Rust, LabVIEW), or edge case? Just say the word!"
        )

    # 6. General Knowledge / Research Q&A (Uses Live Wikipedia Extract + GitHub Sources!)
    if wiki_extract:
        source_links = ""
        if sources:
            source_links = "\n\n#### 🔗 Verified References\n" + "\n".join(
                [f"- **[{s.provider}: {s.title}]({s.url})** — {s.snippet}" for s in sources[:3]]
            )
        return (
            f"### 💡 Dost.ai Answer: *{message}*\n\n"
            f"{wiki_extract}\n"
            f"{source_links}\n\n"
            "Would you like me to dive deeper into any part of this, show a practical example, or write code for it?"
        )

    return (
        f"### 🤝 Dost.ai Personal Assistant\n\n"
        f"Got it! Regarding **\"{message}\"** — here is how we can approach this step by step:\n\n"
        "1. **Clarify the Core Objective:** Tell me the exact outcome or format you want (a quick summary, a schedule, a message draft, or working code).\n"
        "2. **Fast Execution:** I can generate the full draft, calculation, or implementation right here in chat.\n"
        "3. **Iterate Together:** Just reply naturally with any tweaks you want!\n\n"
        "How would you like me to tailor this for you, Dost?"
    )


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "assistant": "Dost.ai",
        "version": "2.2.0",
        "gemini_configured": bool(DEFAULT_GEMINI_KEY),
        "model": GEMINI_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    t0 = time.perf_counter()
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    intent = classify_intent(msg)
    active_key = (req.gemini_key or DEFAULT_GEMINI_KEY or "").strip().strip('"').strip("'")

    sources: List[SourceItem] = []
    wiki_extract = ""

    async with httpx.AsyncClient(trust_env=False, timeout=20.0) as client:
        # Only run external Wikipedia/GitHub search when the user asks a knowledge/code/engineering question
        if intent in ("knowledge_qa", "engineering_lab", "coding_assistant"):
            wiki_task = search_wikipedia(client, msg)
            gh_task = search_github(client, msg)
            (wiki_res, wiki_extract), gh_res = await asyncio.gather(wiki_task, gh_task)
            sources = (wiki_res + gh_res)[:4]

        reply_md = None
        engine_used = "Dost.ai Personal Assistant Brain"

        if active_key:
            reply_md, used_model, _ = await call_gemini_api(client, active_key, msg, req.history, sources)
            if reply_md:
                engine_used = f"Google Gemini ({used_model})"

        if not reply_md:
            reply_md = build_conversational_assistant_response(
                msg, intent, req.history, sources, wiki_extract
            )

    widget = detect_visual_widget(msg, intent)
    suggestions = generate_smart_suggestions(msg, intent)
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
