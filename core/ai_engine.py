"""
ai_engine.py  —  Multi-Provider AI Engine
==========================================
Supports THREE providers. User enters their own key at runtime:

  ┌──────────┬──────────────────────────────┬──────────────────────┐
  │ Provider │ Model                        │ Cost                 │
  ├──────────┼──────────────────────────────┼──────────────────────┤
  │ Groq     │ llama-3.3-70b (+ 3 more)    │ Free                 │
  │ Gemini   │ gemini-2.0-flash (auto)      │ Free                 │
  │ OpenAI   │ gpt-4o-mini / gpt-4o         │ Paid (cheap)         │
  └──────────┴──────────────────────────────┴──────────────────────┘

Keys are NEVER stored on disk. They come from st.session_state
at runtime and are passed into every function call.
"""

import os
import json
import re
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load .env for developer's own keys (optional fallback)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# ── Groq model menu ───────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B  ⚡ Recommended",
    "llama3-70b-8192":         "LLaMA 3 70B    🔥 Fast",
    "mixtral-8x7b-32768":      "Mixtral 8x7B   🌀 Long context",
    "gemma2-9b-it":            "Gemma 2 9B     🪶 Lightweight",
}
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Gemini model priority ─────────────────────────────────────────
GEMINI_PREFERRED = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
    "gemini-1.5-flash", "gemini-1.5-flash-latest",
    "gemini-1.5-pro",   "gemini-pro",
]
_gemini_model_cache = None


# ══════════════════════════════════════════════════════════════════
# LOW-LEVEL API CALLERS  (all use runtime key passed in)
# ══════════════════════════════════════════════════════════════════

def _call_groq(prompt: str, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> str:
    """Direct httpx call to Groq's OpenAI-compatible REST endpoint."""
    if not api_key:
        raise ValueError("Groq API key is missing. Please enter it in the sidebar.")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500,
            },
        )
    if resp.status_code != 200:
        raise ValueError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str, api_key: str) -> str:
    """Direct httpx call to Gemini generateContent REST endpoint."""
    global _gemini_model_cache

    if not api_key:
        raise ValueError("Gemini API key is missing. Please enter it in the sidebar.")

    # Auto-discover best available model (cached after first call)
    if not _gemini_model_cache:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            available = [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            ]
            for preferred in GEMINI_PREFERRED:
                if preferred in available:
                    _gemini_model_cache = preferred
                    break
            else:
                _gemini_model_cache = "gemini-2.0-flash"
        except Exception:
            _gemini_model_cache = "gemini-2.0-flash"

    # Call via REST directly (no SDK dependency for actual call)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model_cache}:generateContent?key={api_key}"
    )
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Direct httpx call to OpenAI chat completions endpoint."""
    if not api_key:
        raise ValueError("OpenAI API key is missing. Please enter it in the sidebar.")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500,
            },
        )
    if resp.status_code != 200:
        raise ValueError(f"OpenAI API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ai(prompt: str, provider: str, api_key: str,
             groq_model: str = DEFAULT_GROQ_MODEL,
             openai_model: str = "gpt-4o-mini") -> str:
    """
    Unified AI router. Calls the right provider with the user's runtime key.

    Args:
        prompt:       Full prompt string
        provider:     "groq" | "gemini" | "openai"
        api_key:      User's runtime key from session_state
        groq_model:   Which Groq model (ignored for other providers)
        openai_model: Which OpenAI model (ignored for other providers)
    """
    if not api_key or len(api_key.strip()) < 10:
        raise ValueError(
            f"No valid API key found for {provider}. "
            "Please paste your key in the sidebar before using AI features."
        )

    p = provider.lower()
    if p == "groq":
        return _call_groq(prompt, api_key.strip(), groq_model)
    elif p == "gemini":
        return _call_gemini(prompt, api_key.strip())
    elif p == "openai":
        return _call_openai(prompt, api_key.strip(), openai_model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'groq', 'gemini', or 'openai'.")


# ══════════════════════════════════════════════════════════════════
# JSON PARSER
# ══════════════════════════════════════════════════════════════════

def _parse_json_safely(text: str):
    """Strip markdown fences and robustly parse JSON from AI response."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r'\{.*\}', r'\[.*?\]']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


# ══════════════════════════════════════════════════════════════════
# FEATURE 1: Polish Professional Summary
# ══════════════════════════════════════════════════════════════════

def polish_summary(raw_summary: str, name: str, job_title: str,
                   provider: str = "groq", api_key: str = "",
                   groq_model: str = DEFAULT_GROQ_MODEL,
                   openai_model: str = "gpt-4o-mini") -> str:
    """
    Detects weak/informal summaries and rewrites them professionally.
    Skips if already well-written (>40 words, no informal phrases).
    """
    if not raw_summary.strip():
        return ""

    bad_phrases = ["i am", "i worked", "i do", "i have", "i was", "i know", "i made"]
    is_weak = len(raw_summary.split()) < 30 or any(b in raw_summary.lower() for b in bad_phrases)
    if not is_weak:
        return raw_summary

    prompt = f"""You are an expert resume writer.

APPLICANT NAME: {name}
JOB TITLE: {job_title}
DRAFT: "{raw_summary}"

Rewrite as a professional ATS-optimized 2-3 sentence summary.
- Third person ("Results-driven Data Analyst with...")
- Confident, professional tone
- Reference the job title; infer expertise from context
- No fake metrics or company names
- Return ONLY the rewritten text. Nothing else.

REWRITTEN SUMMARY:"""

    try:
        result = _call_ai(prompt, provider, api_key, groq_model, openai_model)
        polished = result.strip().strip('"')
        return polished if len(polished) > 20 else raw_summary
    except Exception:
        return raw_summary


# ══════════════════════════════════════════════════════════════════
# FEATURE 2: Enhance Experience Bullet Points
# ══════════════════════════════════════════════════════════════════

def enhance_experience_bullets(raw_description: str, job_title: str,
                                provider: str = "groq", api_key: str = "",
                                groq_model: str = DEFAULT_GROQ_MODEL,
                                openai_model: str = "gpt-4o-mini") -> list:
    """
    Transforms raw experience into 3-5 powerful ATS bullet points.
    """
    if not raw_description.strip():
        return ["No experience description provided."]

    prompt = f"""You are an expert resume writer specializing in ATS optimization.

ROLE: {job_title}
RAW INPUT: "{raw_description}"

Transform into 3-5 powerful ATS resume bullet points.

RULES:
1. Start with strong action verb (Led, Built, Optimized, Reduced, Increased, Developed, Managed, Delivered, Analyzed, Automated, Streamlined, Implemented)
2. Quantify where logical (%, numbers, scale)
3. Focus on impact and outcomes
4. Max 20 words per bullet
5. If input is vague, infer responsibilities from the job title
6. Return ONLY a valid JSON array of strings. No markdown, no explanation.

OUTPUT:"""

    try:
        raw_text = _call_ai(prompt, provider, api_key, groq_model, openai_model)
        result = _parse_json_safely(raw_text)

        if isinstance(result, list):
            bullets = [str(b).strip() for b in result if str(b).strip()]
            return bullets[:5] if bullets else [raw_description]

        lines = [
            line.strip().lstrip('•-*0123456789. "\'')
            for line in raw_text.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]
        return lines[:5] if lines else [raw_description]

    except Exception as e:
        return [f"⚠️ {str(e)[:200]}"]


# ══════════════════════════════════════════════════════════════════
# FEATURE 3: Analyze CV Against Job Description
# ══════════════════════════════════════════════════════════════════

def analyze_cv_against_jd(cv_text: str, job_description: str,
                           provider: str = "groq", api_key: str = "",
                           groq_model: str = DEFAULT_GROQ_MODEL,
                           openai_model: str = "gpt-4o-mini") -> dict:
    """
    Deep ATS analysis: CV vs Job Description.
    Returns score, matched/missing keywords, strengths, suggestions.
    """
    if not cv_text.strip():
        return {"error": "CV text is empty. Could not extract text from the PDF."}
    if not job_description.strip():
        return {"error": "Job description is empty."}
    if not api_key or len(api_key.strip()) < 10:
        return {"error": f"No API key for '{provider}'. Please enter your key in the sidebar."}

    prompt = f"""You are a senior ATS specialist and hiring manager with 15 years of experience.

Analyze the CV against the Job Description. Return ONLY raw JSON — no markdown, no explanation.

{{
  "score": <integer 0-100>,
  "matched_keywords": ["kw1", "kw2"],
  "missing_keywords": ["kw1", "kw2"],
  "strengths": ["s1", "s2", "s3"],
  "suggestions": ["action 1", "action 2", "action 3", "action 4"],
  "summary": "2-3 sentence executive summary of match quality."
}}

Scoring: 85-100 excellent · 70-84 good · 50-69 moderate · 0-49 poor
Be honest — weak CV content should lower the score.

--- CV ---
{cv_text[:4000]}

--- JOB DESCRIPTION ---
{job_description[:3000]}"""

    try:
        raw_text = _call_ai(prompt, provider, api_key, groq_model, openai_model)
        result = _parse_json_safely(raw_text)

        if not isinstance(result, dict):
            raise ValueError("AI response was not valid JSON.")

        defaults = {
            "score": 0, "matched_keywords": [], "missing_keywords": [],
            "strengths": [], "suggestions": [], "summary": "Analysis complete."
        }
        for k, v in defaults.items():
            if k not in result:
                result[k] = v

        result["score"] = max(0, min(100, int(result.get("score", 0))))
        return result

    except Exception as e:
        return {
            "error": str(e)[:300],
            "score": 0, "matched_keywords": [], "missing_keywords": [],
            "strengths": [], "suggestions": [], "summary": "Analysis failed."
        }
