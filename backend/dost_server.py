"""
Dost.ai v3.0 — Antigravity-Style Personal AI Assistant, Live Tab Tracker & Visual Search Engine
Powered by FastAPI + Google Gemini API + Stateful Agent Memory, Tasks & Live Telemetry Watchers.
"""

import os
import re
import time
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Dost.ai Antigravity Personal Assistant API",
    description="Stateful Agentic Personal Assistant that keeps tabs on Tasks, Memory, Portfolio Telemetry & GitHub",
    version="3.0.0",
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
LAST_GEMINI_DIAG = "Not called yet"

# Persistent server-side store (also mirrored in browser localStorage)
STATE_FILE = Path(__file__).resolve().parent / "dost_state.json"

DEFAULT_STATE: Dict[str, Any] = {
    "tasks": [
        {
            "id": "t-1",
            "title": "Review Ducom Aerospace Tribology DAQ & Vision pipeline telemetry",
            "category": "Engineering",
            "priority": "HIGH",
            "status": "active",
            "due": "Today · 17:00",
            "created_at": "2026-11-07",
        },
        {
            "id": "t-2",
            "title": "Verify OmniComm Studio USB camera & Modbus RTU fieldbus packets",
            "category": "Open Source",
            "priority": "MEDIUM",
            "status": "active",
            "due": "Today · 19:30",
            "created_at": "2026-11-07",
        },
        {
            "id": "t-3",
            "title": "Check portfolio visitor logs & recruiter outreach messages",
            "category": "Career & Network",
            "priority": "HIGH",
            "status": "active",
            "due": "Daily · 20:00",
            "created_at": "2026-11-07",
        },
    ],
    "memories": [
        {
            "id": "m-1",
            "topic": "Owner Profile",
            "content": "Kiran Shivakumar — Application Engineer (Mechatronics, LabVIEW, QMH/DQMH, Python, ROS, Computer Vision/YOLO) based in Bengaluru.",
            "timestamp": "Pinned",
        },
        {
            "id": "m-2",
            "topic": "Portfolio Telemetry & PIN",
            "content": "Portfolio URL: https://kiran10299.github.io/ | Owner Console Shortcut: Ctrl+Shift+L or ?owner=kiran (PIN: 1029) | Stream: ntfy.sh/kiran_shivakumar_portfolio_log_99a7",
            "timestamp": "Pinned",
        },
        {
            "id": "m-3",
            "topic": "Key Open-Source Flagships",
            "content": "1) Dost.ai (dost-ai-personal-assistant.onrender.com), 2) OmniComm Studio (Universal-Device-Communicator), 3) OpenATE-Engine, 4) EdgeVibe-Pro.",
            "timestamp": "Pinned",
        },
    ],
    "updates": [
        {
            "id": "u-1",
            "source": "Dost.ai Core",
            "title": "Antigravity Multi-Tab Agent v3.0 Online",
            "detail": "Actively keeping tabs on your Tasks, Memory Vault, GitHub Repos, and Portfolio Visitors.",
            "time": "Just now",
            "type": "system",
        }
    ],
}


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_STATE))


def save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


SERVER_STATE = load_state()


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ClientStateSync(BaseModel):
    tasks: Optional[List[Dict[str, Any]]] = None
    memories: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or command for Dost.ai")
    history: List[ChatMessage] = Field(default_factory=list)
    gemini_key: Optional[str] = Field(default=None)
    mode: Optional[str] = Field(default="auto")
    client_state: Optional[ClientStateSync] = Field(default=None)


class SourceItem(BaseModel):
    provider: str
    title: str
    snippet: str
    url: str
    thumbnail: Optional[str] = None
    badge: Optional[str] = None


class AgentStep(BaseModel):
    icon: str
    tool: str
    summary: str
    status: str = "done"


class VisualWidget(BaseModel):
    widget_type: str
    title: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    assistant_name: str = "Dost.ai"
    engine_used: str
    agent_steps: List[AgentStep] = Field(default_factory=list)
    reply_markdown: str
    sources: List[SourceItem] = Field(default_factory=list)
    visual_widget: Optional[VisualWidget] = None
    updated_state: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    latency_ms: int


def classify_intent(query: str) -> str:
    q = query.lower().strip()

    # 1. Explicit Tab / Task / Reminder Management
    if any(p in q for p in [
        "add task", "new task", "keep a tab on", "keep tab on", "track this",
        "remind me to", "remind me about", "add to my tabs", "put on my todo",
        "add todo", "schedule task", "create task"
    ]):
        return "add_task"

    if any(p in q for p in [
        "complete task", "mark done", "finished task", "clear completed",
        "check off"
    ]):
        return "complete_task"

    # 2. Memory / Note Saving ("remember that...", "save note...", "keep in mind...")
    if any(p in q for p in [
        "remember that", "remember this", "save note", "add note",
        "keep in mind", "memorize", "store this", "note down"
    ]):
        return "save_memory"

    # 3. Status Update / Briefing / What's on my tabs / Portfolio / GitHub check
    if any(p in q for p in [
        "status update", "update me", "brief me", "briefing", "what tabs",
        "my tabs", "what is pending", "what's pending", "my tasks",
        "what do i have today", "portfolio visitor", "who visited",
        "who viewed my portfolio", "github status", "my repos", "radar",
        "what did you remember", "my notes", "my memory"
    ]):
        return "status_briefing"

    # 4. Hardware / Signal / Circuit Lab
    if any(k in q for k in [
        "4-20", "4 to 20", "current loop", "rc filter", "low pass", "low-pass",
        "cutoff frequency", "bode", "ohm's law", "ohms law", "voltage divider",
        "oscilloscope", "pwm", "fourier", "harmonic", "nyquist", "shunt resistor"
    ]):
        return "engineering_lab"

    # 5. Casual / Greeting / Personal Conversation
    words = re.findall(r"\b\w+\b", q)
    greetings = [
        "hi", "hello", "hey", "hii", "heyy", "yo", "namaste", "good morning",
        "good afternoon", "good evening", "good night", "how are you", "how r u",
        "what's up", "whats up", "sup", "who are you", "what is your name",
        "whats your name", "are you there", "can we talk", "talk to me",
        "thank you", "thanks", "cool", "awesome", "great", "nice", "okay", "ok",
        "tell me a joke", "motivate me", "i am tired", "i'm tired", "i am bored",
        "i feel stressed", "help me", "what can you do"
    ]
    if q in greetings or (len(words) <= 7 and any(g in q for g in greetings)):
        return "casual_chat"

    # 6. Daily Planning
    if any(k in q for k in [
        "plan my day", "my schedule", "daily routine", "time table", "timetable",
        "organize my", "productive", "study plan", "prioritize", "manage my time"
    ]):
        return "daily_planner"

    # 7. Writing / Emails
    if any(k in q for k in [
        "write an email", "draft an email", "email to", "linkedin post",
        "cover letter", "message to", "reply to"
    ]):
        return "writing_assistant"

    # 8. Coding
    if any(k in q for k in [
        "write code", "python", "javascript", "typescript", "c++", "rust",
        "fastapi", "react", "kotlin", "android", "sql", "algorithm", "debug", "labview"
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
        "where", "when", "today", "doing", "help", "my", "your", "update", "tabs"
    }
    cleaned = re.sub(r"[^\w\s\-\.]", " ", query)
    words = [w for w in cleaned.split() if w.lower() not in stop_words and len(w) > 1]
    return " ".join(words[:4]) if words else ""


async def fetch_live_portfolio_visitors(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    url = "https://ntfy.sh/kiran_shivakumar_portfolio_log_99a7/json?poll=1&since=48h"
    visitors = []
    try:
        resp = await client.get(url, timeout=3.5)
        if resp.status_code == 200:
            for line in resp.text.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "message" and ev.get("message"):
                        payload = json.loads(ev["message"])
                        visitors.append(payload)
                except Exception:
                    continue
    except Exception:
        pass
    return visitors[-5:][::-1]


async def fetch_kiran_github_repos(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    url = "https://api.github.com/users/kiran10299/repos?sort=updated&per_page=5"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DostAI-Agent/3.0"}
    try:
        resp = await client.get(url, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            items = resp.json()
            return [
                {
                    "name": r.get("name"),
                    "url": r.get("html_url"),
                    "stars": r.get("stargazers_count", 0),
                    "language": r.get("language") or "Multi",
                    "updated_at": (r.get("pushed_at") or "")[:10],
                    "description": (r.get("description") or "")[:120],
                }
                for r in items[:5]
            ]
    except Exception:
        pass
    return []


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
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DostAI-Agent/3.0"}
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
            clean_desc = re.sub(r"\s+", " ", raw_desc)[:150]
            if len(raw_desc) > 150:
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


def build_agent_system_prompt(state: Dict[str, Any], visitors: List[Dict[str, Any]], gh_repos: List[Dict[str, Any]]) -> str:
    active_tasks = [t for t in state.get("tasks", []) if t.get("status") != "done"]
    tasks_str = "\n".join([f"  - [{t.get('priority','MED')}] {t.get('title')} (Due: {t.get('due','Soon')})" for t in active_tasks]) or "  - No pending tasks."
    mem_str = "\n".join([f"  - {m.get('topic')}: {m.get('content')}" for m in state.get("memories", [])[-8:]]) or "  - None."
    vis_str = "\n".join([f"  - {v.get('timeStr','Recent')} from {v.get('location','Unknown')} ({v.get('org','ISP')}) -> {v.get('action','Viewed Portfolio')}" for v in visitors[:3]]) or "  - No recent visitor pings in last 48h window."
    repo_str = "\n".join([f"  - {r.get('name')} (★ {r.get('stars')}, Last push: {r.get('updated_at')})" for r in gh_repos[:4]]) or "  - Synced with github.com/kiran10299"

    return f"""You are **Dost.ai (v3.0)** — an Antigravity-class Agentic Personal Assistant, Chief-of-Staff, and Engineering Co-Pilot created for **Kiran Shivakumar**.
"Dost" means trusted best friend and companion. You proactively keep tabs on Kiran's tasks, projects, memories, portfolio visitors, and GitHub repositories, and update him conversationally.

### LIVE TABS & STATE YOU ARE CURRENTLY KEEPING FOR KIRAN:
1. **ACTIVE TASKS TAB ({len(active_tasks)} Open):**
{tasks_str}

2. **MEMORY & NOTES VAULT:**
{mem_str}

3. **PORTFOLIO VISITOR RADAR (Live Telemetry):**
{vis_str}

4. **GITHUB REPOSITORIES WATCHER (@kiran10299):**
{repo_str}

### BEHAVIOR GUIDELINES:
- Speak naturally, warmly, and proactively like an elite Antigravity AI agent + loyal friend ("Dost").
- Whenever Kiran asks for a status update, briefing, or how things are going, reference his actual open tasks, memories, portfolio visitors, and GitHub activity above!
- Never dump unrelated physics formulas on casual or personal assistant queries. Reserve LaTeX math (`$$...$$`) and circuit details for when Kiran asks an engineering/math question.
"""


async def call_gemini_api(
    client: httpx.AsyncClient,
    api_key: str,
    message: str,
    history: List[ChatMessage],
    sources: List[SourceItem],
    sys_prompt: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    global LAST_GEMINI_DIAG
    clean_key = api_key.strip().strip('"').strip("'")
    grounding_text = "\n".join(
        [f"- [{s.provider}] {s.title}: {s.snippet[:150]}" for s in sources[:3]]
    )
    contents = []
    for h in history[-8:]:
        role = "user" if h.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.content[:1500]}]})

    user_text = message
    if grounding_text:
        user_text += f"\n\n[Live Search Grounding]:\n{grounding_text}"

    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "systemInstruction": {"parts": [{"text": sys_prompt}]},
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
        "gemini-1.5-flash-latest",
        "gemini-pro",
    ]
    last_err = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
        try:
            resp = await client.post(url, json=payload, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    if text.strip():
                        LAST_GEMINI_DIAG = f"OK ({model_name})"
                        return text, model_name, None
            else:
                err_snip = resp.text[:160].replace("\n", " ")
                last_err = f"{model_name} HTTP {resp.status_code}: {err_snip}"
        except Exception as e:
            last_err = f"{model_name}: {type(e).__name__}"
            continue
    LAST_GEMINI_DIAG = last_err or "Failed"
    return None, None, last_err


async def call_free_cloud_llm(
    client: httpx.AsyncClient,
    message: str,
    history: List[ChatMessage],
    sys_prompt: str,
) -> Optional[str]:
    """Zero-key cloud LLM fallback so Dost.ai can converse freely on any topic even without a Gemini key."""
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-6:]:
        messages.append({"role": "user" if h.role == "user" else "assistant", "content": h.content[:1000]})
    messages.append({"role": "user", "content": message})

    try:
        resp = await client.post(
            "https://text.pollinations.ai/openai",
            json={"model": "openai", "messages": messages, "temperature": 0.65},
            timeout=9.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                txt = (choices[0].get("message", {}).get("content") or "").strip()
                if txt and len(txt) > 10:
                    return txt
    except Exception:
        pass
    return None


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "assistant": "Dost.ai",
        "version": "3.0.0",
        "gemini_configured": bool(DEFAULT_GEMINI_KEY),
        "gemini_key_prefix": (DEFAULT_GEMINI_KEY[:6] + "...") if DEFAULT_GEMINI_KEY else "none",
        "last_gemini_diag": LAST_GEMINI_DIAG,
        "active_tasks_count": len([t for t in SERVER_STATE.get("tasks", []) if t.get("status") != "done"]),
        "memories_count": len(SERVER_STATE.get("memories", [])),
    }


@app.get("/api/state")
async def get_state_endpoint():
    async with httpx.AsyncClient(trust_env=False, timeout=5.0) as client:
        vis_task = fetch_live_portfolio_visitors(client)
        gh_task = fetch_kiran_github_repos(client)
        visitors, gh_repos = await asyncio.gather(vis_task, gh_task)
    return {
        "state": SERVER_STATE,
        "live_visitors": visitors,
        "github_repos": gh_repos,
    }


@app.post("/api/state")
async def update_state_endpoint(sync: ClientStateSync):
    if sync.tasks is not None:
        SERVER_STATE["tasks"] = sync.tasks
    if sync.memories is not None:
        SERVER_STATE["memories"] = sync.memories
    save_state(SERVER_STATE)
    return {"status": "synced", "state": SERVER_STATE}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    t0 = time.perf_counter()
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Sync client state if sent
    if req.client_state:
        if req.client_state.tasks is not None:
            SERVER_STATE["tasks"] = req.client_state.tasks
        if req.client_state.memories is not None:
            SERVER_STATE["memories"] = req.client_state.memories

    intent = classify_intent(msg)
    agent_steps: List[AgentStep] = []
    sources: List[SourceItem] = []
    wiki_extract = ""

    # 1. Execute Agentic Tab Mutations if user asked to add a task, complete a task, or save a memory
    if intent == "add_task":
        clean_title = re.sub(
            r"^(hey dost|dost|please|can you|keep a tab on|keep tab on|add task|new task|remind me to|remind me about|track this|add to my tabs|put on my todo|add todo)\s*[:,\-]?\s*",
            "",
            msg,
            flags=re.IGNORECASE,
        ).strip()
        if not clean_title:
            clean_title = msg
        new_task = {
            "id": f"t-{int(time.time() * 1000)}",
            "title": clean_title[0].upper() + clean_title[1:],
            "category": "Personal Tab",
            "priority": "HIGH" if "urgent" in msg.lower() or "important" in msg.lower() else "MEDIUM",
            "status": "active",
            "due": "Tracked by Dost.ai",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        SERVER_STATE["tasks"].insert(0, new_task)
        SERVER_STATE["updates"].insert(
            0,
            {
                "id": f"u-{int(time.time() * 1000)}",
                "source": "Tasks Tab",
                "title": f"Added New Tab: {new_task['title']}",
                "detail": f"Priority: {new_task['priority']} · Added via Dost Agent chat",
                "time": datetime.now().strftime("%H:%M"),
                "type": "task",
            },
        )
        save_state(SERVER_STATE)
        agent_steps.append(
            AgentStep(
                icon="📌",
                tool="update_active_tabs",
                summary=f"Created active tab & task: '{new_task['title']}'",
            )
        )

    elif intent == "save_memory":
        clean_mem = re.sub(
            r"^(hey dost|dost|please|can you|remember that|remember this|save note|add note|keep in mind|memorize|store this|note down)\s*[:,\-]?\s*",
            "",
            msg,
            flags=re.IGNORECASE,
        ).strip()
        if not clean_mem:
            clean_mem = msg
        new_mem = {
            "id": f"m-{int(time.time() * 1000)}",
            "topic": "User Note",
            "content": clean_mem[0].upper() + clean_mem[1:],
            "timestamp": datetime.now().strftime("%b %d, %H:%M"),
        }
        SERVER_STATE["memories"].insert(0, new_mem)
        SERVER_STATE["updates"].insert(
            0,
            {
                "id": f"u-{int(time.time() * 1000)}",
                "source": "Memory Vault",
                "title": "Saved New Note to Memory Vault",
                "detail": new_mem["content"][:90],
                "time": datetime.now().strftime("%H:%M"),
                "type": "memory",
            },
        )
        save_state(SERVER_STATE)
        agent_steps.append(
            AgentStep(
                icon="🧠",
                tool="commit_to_memory_vault",
                summary=f"Stored in persistent Memory Vault: '{new_mem['content'][:60]}'",
            )
        )

    elif intent == "complete_task":
        active_tasks = [t for t in SERVER_STATE.get("tasks", []) if t.get("status") != "done"]
        if active_tasks:
            active_tasks[0]["status"] = "done"
            save_state(SERVER_STATE)
            agent_steps.append(
                AgentStep(
                    icon="✅",
                    tool="complete_active_tab",
                    summary=f"Marked task completed: '{active_tasks[0]['title']}'",
                )
            )

    async with httpx.AsyncClient(trust_env=False, timeout=18.0) as client:
        # Fetch live telemetry when briefing or checking status
        visitors: List[Dict[str, Any]] = []
        gh_repos: List[Dict[str, Any]] = []

        if intent in ("status_briefing", "casual_chat", "daily_planner"):
            vis_task = fetch_live_portfolio_visitors(client)
            gh_task = fetch_kiran_github_repos(client)
            visitors, gh_repos = await asyncio.gather(vis_task, gh_task)
            agent_steps.append(
                AgentStep(
                    icon="📡",
                    tool="scan_live_tabs_and_telemetry",
                    summary=f"Scanned {len([t for t in SERVER_STATE['tasks'] if t.get('status')!='done'])} active tasks, {len(SERVER_STATE['memories'])} memories & live GitHub/Portfolio feeds",
                )
            )

        if intent in ("knowledge_qa", "engineering_lab", "coding_assistant"):
            wiki_task = search_wikipedia(client, msg)
            gh_s_task = search_github(client, msg)
            (wiki_res, wiki_extract), gh_res = await asyncio.gather(wiki_task, gh_s_task)
            sources = (wiki_res + gh_res)[:4]
            if sources:
                agent_steps.append(
                    AgentStep(
                        icon="🌐",
                        tool="multi_source_search",
                        summary=f"Retrieved {len(sources)} live citations from Wikipedia & GitHub",
                    )
                )

        sys_prompt = build_agent_system_prompt(SERVER_STATE, visitors, gh_repos)
        active_key = (req.gemini_key or DEFAULT_GEMINI_KEY or "").strip().strip('"').strip("'")

        reply_md = None
        engine_used = "Dost.ai Antigravity Agent"

        # 1. Try Google Gemini API
        if active_key:
            reply_md, used_model, _ = await call_gemini_api(
                client, active_key, msg, req.history, sources, sys_prompt
            )
            if reply_md:
                engine_used = f"Google Gemini ({used_model}) · Agent Mode"
                agent_steps.append(
                    AgentStep(
                        icon="✨",
                        tool=f"gemini_generate ({used_model})",
                        summary="Synthesized context-aware response with live tab state",
                    )
                )

        # 2. Try Free Cloud Conversational LLM if Gemini didn't return
        if not reply_md and intent not in ("add_task", "save_memory", "status_briefing"):
            cloud_reply = await call_free_cloud_llm(client, msg, req.history, sys_prompt)
            if cloud_reply:
                reply_md = cloud_reply
                engine_used = "Dost.ai Cloud LLM · Agent Mode"
                agent_steps.append(
                    AgentStep(
                        icon="⚡",
                        tool="cloud_llm_synthesis",
                        summary="Generated conversational agent response with live tab memory",
                    )
                )

        # 3. Deterministic Stateful Agent Synthesis (instant, rich, and 100% aware of all open tabs)
        if not reply_md:
            reply_md = build_stateful_agent_markdown(
                msg, intent, SERVER_STATE, visitors, gh_repos, sources, wiki_extract
            )

    widget = None
    if intent == "engineering_lab":
        q_low = msg.lower()
        if any(k in q_low for k in ["4-20", "current loop", "plc", "shunt"]):
            widget = VisualWidget(
                widget_type="circuit_calc",
                title="Interactive 4–20mA Current Loop & ADC Scaling Lab",
                params={"mode": "4_20ma", "current_ma": 12.0, "shunt_ohm": 250.0},
            )
        elif any(k in q_low for k in ["rc ", "rc-", "low pass", "cutoff", "filter", "bode"]):
            widget = VisualWidget(
                widget_type="circuit_calc",
                title="Interactive RC Low-Pass Filter & Cutoff Frequency Lab",
                params={"mode": "rc_filter", "resistor_ohm": 1000.0, "cap_nf": 100.0},
            )
        else:
            widget = VisualWidget(
                widget_type="oscilloscope",
                title="Dost.ai Real-Time Interactive Signal & Harmonic Oscilloscope",
                params={"waveform": "harmonic", "freq_hz": 4.0, "noise": 0.05},
            )

    suggestions = [
        "Give me a full status briefing on all my open tabs",
        "Keep a tab on finishing the LabVIEW driver by 6 PM",
        "Remember that my test bench PLC IP is 192.168.1.50",
        "Who visited my portfolio recently?",
    ]
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return ChatResponse(
        assistant_name="Dost.ai",
        engine_used=engine_used,
        agent_steps=agent_steps,
        reply_markdown=reply_md,
        sources=sources,
        visual_widget=widget,
        updated_state=SERVER_STATE,
        suggestions=suggestions,
        latency_ms=latency_ms,
    )


def build_stateful_agent_markdown(
    message: str,
    intent: str,
    state: Dict[str, Any],
    visitors: List[Dict[str, Any]],
    gh_repos: List[Dict[str, Any]],
    sources: List[SourceItem],
    wiki_extract: str,
) -> str:
    active_tasks = [t for t in state.get("tasks", []) if t.get("status") != "done"]
    memories = state.get("memories", [])

    if intent == "add_task":
        latest = state["tasks"][0]
        return (
            f"### 📌 Tab Added & Tracked: **{latest['title']}**\n\n"
            f"I've locked this into your **📋 Active Tasks & Projects** tab (`Priority: {latest['priority']}`).\n\n"
            f"You now have **{len(active_tasks)} active tabs** I'm tracking for you:\n"
            + "\n".join([f"- **[{t.get('priority','MED')}]** {t.get('title')} *(Due: {t.get('due','Soon')})*" for t in active_tasks[:5]])
            + "\n\nI'll keep watching this and include it in your briefings. Want to set a specific time or add any notes to it?"
        )

    if intent == "save_memory":
        latest_m = state["memories"][0]
        return (
            f"### 🧠 Committed to Dost Memory Vault\n\n"
            f"> **\"{latest_m['content']}\"**\n\n"
            f"I've saved this in your **🧠 Memory & Notes Vault** ({len(memories)} total memories stored). Whenever you ask about it in future conversations, I'll recall it automatically!"
        )

    if intent == "complete_task":
        return (
            f"### ✅ Task Checked Off!\n\n"
            f"I've updated your **📋 Active Tasks** tab. You now have **{len(active_tasks)} remaining active tasks**.\n\n"
            + ("\n".join([f"- **[{t.get('priority','MED')}]** {t.get('title')}" for t in active_tasks[:4]]) if active_tasks else "🎉 **All active tasks are cleared!**")
        )

    if intent == "status_briefing":
        tasks_md = "\n".join(
            [f"- **[{t.get('priority','MED')}] {t.get('title')}** — *{t.get('category','Task')} · {t.get('due','Active')}*" for t in active_tasks[:6]]
        ) or "- 🎉 All tasks completed! Tell me *\"Keep a tab on...\"* to add a new one."

        mem_md = "\n".join(
            [f"- **{m.get('topic','Note')}:** {m.get('content')}" for m in memories[:4]]
        ) or "- No saved notes yet."

        vis_md = "\n".join(
            [f"- **{v.get('timeStr','Recent')}** · **{v.get('location','Global Visitor')}** (`{v.get('org','Network')}`) — *{v.get('action','Viewed Portfolio')}*" for v in visitors[:4]]
        ) or "- Listening on `ntfy.sh/kiran_shivakumar_portfolio_log_99a7` (Open your portfolio to trigger a live ping!)."

        gh_md = "\n".join(
            [f"- **[`{r.get('name')}`]({r.get('url')})** ({r.get('language')}) — ★ {r.get('stars')} · Last push: `{r.get('updated_at')}`" for r in gh_repos[:4]]
        ) or "- Connected to `@kiran10299` GitHub profile."

        return (
            f"### ⚡ Dost.ai Executive Briefing & Active Tabs Report\n\n"
            f"Here is the live status across all the tabs I'm monitoring for you right now, Kiran:\n\n"
            f"#### 📋 1. Active Tasks & Open Tabs ({len(active_tasks)} Pending)\n"
            f"{tasks_md}\n\n"
            f"#### 👁️ 2. Portfolio Visitor Radar (Live Telemetry)\n"
            f"{vis_md}\n\n"
            f"#### 🐙 3. GitHub Repositories Activity (`@kiran10299`)\n"
            f"{gh_md}\n\n"
            f"#### 🧠 4. Pinned in Memory Vault ({len(memories)} Items)\n"
            f"{mem_md}\n\n"
            "> 💡 **Quick Agent Commands:** Say *\"Keep a tab on [task]\"*, *\"Remember that [fact]\"*, or switch between the **📋 Active Tasks**, **📡 Live Radar**, and **🧠 Memory Vault** tabs at the top!"
        )

    if intent == "casual_chat":
        top_task = active_tasks[0]["title"] if active_tasks else "your engineering & open-source projects"
        return (
            f"Hey Kiran! 👋 Your **Dost** is online, locked in, and keeping tabs on everything for you!\n\n"
            f"**Quick Pulse Check:**\n"
            f"- 📋 **Active Tasks Open:** **{len(active_tasks)}** *(Top priority: \"{top_task}\")*\n"
            f"- 🧠 **Memory Vault:** **{len(memories)}** notes & configs stored\n"
            f"- 📡 **Live Watchers:** Monitoring your GitHub (`@kiran10299`) & Portfolio Visitor Telemetry\n\n"
            f"How's your day going? Tell me what you want to do — you can say:\n"
            f"- *\"Give me a full status briefing on my tabs\"*\n"
            f"- *\"Keep a tab on [new task or reminder]\"*\n"
            f"- *\"Remember that [important note, IP, or idea]\"*\n"
            f"- Or ask me anything — code, circuits, emails, or casual chat!"
        )

    if wiki_extract:
        ref_md = "\n".join([f"- **[{s.provider}: {s.title}]({s.url})** — {s.snippet}" for s in sources[:3]])
        return (
            f"### 💡 {message}\n\n"
            f"{wiki_extract}\n\n"
            + (f"#### 🔗 Live Sources\n{ref_md}\n\n" if ref_md else "")
            + "Want me to save a summary of this into your **🧠 Memory Vault** or create an action item in your **📋 Active Tasks** tab?"
        )

    # Check if user's question matches anything in Memory Vault
    q_words = [w.lower() for w in re.findall(r"\b\w{3,}\b", message)]
    matched_mems = [
        m for m in memories
        if any(w in (m.get("content", "") + " " + m.get("topic", "")).lower() for w in q_words)
    ]
    if matched_mems:
        m_list = "\n".join([f"- **{m['topic']}:** {m['content']}" for m in matched_mems])
        return (
            f"### 🧠 Found in Your Dost Memory Vault\n\n"
            f"Here is what I have on file regarding **\"{message}\"**:\n\n"
            f"{m_list}\n\n"
            f"Would you like me to update this note or add a follow-up task to your **📋 Active Tasks** tab?"
        )

    return (
        f"### ⚡ Dost.ai Agent Response: *{message}*\n\n"
        f"I'm on it, Kiran! Here is how I've processed this alongside your **{len(active_tasks)} active tabs**:\n\n"
        f"1. **Actionable Next Step:** Tell me if you want me to track **\"{message[:60]}\"** as an active tab in your **📋 Active Tasks** board (just say *\"Keep a tab on {message[:40]}\"*).\n"
        f"2. **Deep Execution:** If you need code, a drafted message, a schedule, or a calculation for this, tell me the details and I'll generate it right now.\n\n"
        f"What's your call, Dost?"
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
        content="<h1>Dost.ai Antigravity Personal Assistant API v3.0 is Online</h1>"
    )
