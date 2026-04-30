"""
app.py
======
ATS CV Builder & Analyzer – Main Streamlit Application

Run with:
    streamlit run app.py

Modules:
    Tab 1 → Interactive CV Builder (AI-enhanced bullet points + PDF export)
    Tab 2 → CV Analyzer & Scorer (Upload PDF + paste JD → ATS score)

Providers:
    Groq  (llama-3.3-70b, mixtral, gemma — free & fast)
    Gemini (gemini-2.0-flash — Google AI Studio free tier)
"""

import streamlit as st
import PyPDF2
import io
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly from project root — must happen before AI imports
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from core.ai_engine import (
    enhance_experience_bullets, analyze_cv_against_jd, polish_summary,
    GROQ_MODELS, DEFAULT_GROQ_MODEL
)
from core.pdf_maker import generate_cv_pdf

# ──────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ATS CV Pro | AI-Powered Resume Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS – Premium SaaS Look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Main Container ── */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1100px;
    }

    /* ── Hero Title ── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ── Card Containers ── */
    .cv-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.3s ease;
    }
    .cv-card:hover {
        border-color: rgba(167,139,250,0.4);
    }

    /* ── Section Labels ── */
    .section-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #7c3aed22, #3b82f622);
        border: 1px solid rgba(124,58,237,0.4);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #a78bfa;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* ── Form Inputs – COMPLETE FIX for text visibility ── */
    /* Target all possible Streamlit input selectors across versions */
    .stTextInput input,
    .stTextInput > div > div > input,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    .stTextArea textarea,
    .stTextArea > div > div > textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #1e1b4b !important;
        background-color: #1e1b4b !important;
        border: 1px solid rgba(124,58,237,0.45) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        caret-color: #a78bfa !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* Placeholder text */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Focus state */
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.25) !important;
        outline: none !important;
        background: #1e1b4b !important;
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
    }

    /* Input wrapper backgrounds — ALL nested divs */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="input"] > div > div,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] > div,
    .stTextInput > div,
    .stTextInput > div > div,
    .stTextArea > div,
    .stTextArea > div > div {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Active/focused wrapper — keep dark */
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="textarea"]:focus-within,
    .stTextInput > div:focus-within,
    .stTextArea > div:focus-within {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Labels */
    .stTextInput label, .stTextArea label,
    .stSelectbox label, .stFileUploader label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #a5b4fc !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 4px !important;
    }

    /* Autofill fix – prevents browser autofill from turning input white */
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus,
    input:-webkit-autofill:active {
        -webkit-box-shadow: 0 0 0 40px #1e1b4b inset !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        caret-color: #a78bfa !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #0d9488) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.55rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(5,150,105,0.35) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(5,150,105,0.5) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(124,58,237,0.4) !important;
    }

    /* ── Score Display ── */
    .score-circle {
        text-align: center;
        padding: 2rem;
    }
    .score-number {
        font-family: 'DM Serif Display', serif;
        font-size: 5rem;
        line-height: 1;
        margin: 0;
    }
    .score-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Keyword Pills ── */
    .keyword-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px;
    }
    .pill-green {
        background: rgba(52,211,153,0.15);
        border: 1px solid rgba(52,211,153,0.4);
        color: #34d399;
    }
    .pill-red {
        background: rgba(248,113,113,0.15);
        border: 1px solid rgba(248,113,113,0.4);
        color: #f87171;
    }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e2e8f0 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Alert / Warning / Error / Success / Info boxes ── */
    /* Error - red */
    div[data-testid="stAlert"][kind="error"],
    .stAlert[data-baseweb="notification"][kind="error"],
    div[role="alert"].stAlert,
    [data-testid="stNotification"] {
        border-radius: 12px !important;
    }
    /* Target ALL streamlit alert types with explicit text color */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    /* Error box specifically */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 12px !important;
    }
    /* Streamlit uses data-baseweb=notification for all alerts */
    [data-baseweb="notification"] {
        border-radius: 12px !important;
    }
    [data-baseweb="notification"] p,
    [data-baseweb="notification"] span,
    [data-baseweb="notification"] li,
    [data-baseweb="notification"] div:not([class*="Icon"]) {
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    /* stAlert wrapper */
    .stAlert {
        border-radius: 12px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    /* NOTE: Do NOT use [data-testid="stSidebar"] * — it overrides everything including badges */
    .sidebar-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
        margin-bottom: 4px;
    }
    .sidebar-step {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 7px 12px;
        margin: 5px 0;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* ── Section Badge (main content area) ── */
    .section-badge {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.18)) !important;
        border: 1px solid rgba(124,58,237,0.5) !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #c4b5fd !important;
        -webkit-text-fill-color: #c4b5fd !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-bottom: 1rem !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(124,58,237,0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

    /* ── Global text color for main content (NOT sidebar) ── */
    .main p, .main span, .main li, .main div:not([class*="stSidebar"]) {
        color: #e2e8f0;
    }

    /* ── CV Card inner text ── */
    .cv-card p, .cv-card span, .cv-card div {
        color: #e2e8f0 !important;
    }

    /* ── st.markdown plain text in main area ── */
    [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        -webkit-text-fill-color: #e2e8f0 !important;
    }

    /* ── Hero subtitle ── */
    .hero-sub {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    /* ── Success / Warning / Error explicit text ── */
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    /* ── st.caption ── */
    [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }

    /* ── Bullet points from st.markdown ── */
    [data-testid="stMarkdownContainer"] ul li,
    [data-testid="stMarkdownContainer"] ol li {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Sidebar — Provider Selector + Instructions
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Sidebar — Runtime API Key Input + Provider Selector
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ ATS CV Pro</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.82rem;margin-top:2px;">AI-Powered Resume Intelligence</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── PROVIDER SELECTOR ────────────────────────────────────────
    st.markdown(
        '<p style="color:#a78bfa;font-weight:700;font-size:0.85rem;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:8px;">🤖 AI Provider</p>',
        unsafe_allow_html=True
    )

    selected_provider = st.radio(
        "provider",
        options=["groq", "gemini", "openai"],
        format_func=lambda x: {
            "groq":   "⚡ Groq  (Free · Fastest)",
            "gemini": "🔷 Google Gemini  (Free)",
            "openai": "🟢 OpenAI  (GPT-4o)",
        }[x],
        label_visibility="collapsed",
        key="ai_provider"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RUNTIME API KEY INPUT ────────────────────────────────────
    # Key info per provider
    provider_meta = {
        "groq": {
            "label":       "Groq API Key",
            "placeholder": "gsk_...",
            "link":        "https://console.groq.com/keys",
            "link_text":   "Get free Groq key →",
            "hint":        "Free · No credit card · Instant",
            "sess_key":    "runtime_groq_key",
        },
        "gemini": {
            "label":       "Gemini API Key",
            "placeholder": "AIzaSy...",
            "link":        "https://aistudio.google.com/app/apikey",
            "link_text":   "Get free Gemini key →",
            "hint":        "Free · Google AI Studio",
            "sess_key":    "runtime_gemini_key",
        },
        "openai": {
            "label":       "OpenAI API Key",
            "placeholder": "sk-...",
            "link":        "https://platform.openai.com/api-keys",
            "link_text":   "Get OpenAI key →",
            "hint":        "Paid · GPT-4o mini recommended",
            "sess_key":    "runtime_openai_key",
        },
    }

    meta = provider_meta[selected_provider]

    # Key input box — password type so it's hidden
    user_key = st.text_input(
        meta["label"],
        type="password",
        placeholder=meta["placeholder"],
        key=meta["sess_key"],
        help="Your key is only stored in this browser session. It is never saved to disk or any server."
    )

    # Status badge
    key_valid = user_key and len(user_key.strip()) > 10

    if key_valid:
        masked = user_key.strip()
        display = f"{masked[:4]}···{masked[-4:]}"
        st.markdown(
            f'<div style="background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.35);'
            f'border-radius:8px;padding:7px 12px;margin:4px 0 8px;">'
            f'<span style="color:#34d399;font-size:0.8rem;font-weight:700;">✅ Key active</span> '
            f'<span style="color:#64748b;font-size:0.75rem;font-family:monospace;">{display}</span><br>'
            f'<span style="color:#475569;font-size:0.72rem;">Session only · clears on browser close</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        # Clear key button
        if st.button("🗑️ Remove Key", key="clear_key_btn"):
            st.session_state[meta["sess_key"]] = ""
            st.rerun()
    else:
        st.markdown(
            f'<div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);'
            f'border-radius:8px;padding:7px 12px;margin:4px 0 8px;">'
            f'<span style="color:#fbbf24;font-size:0.8rem;font-weight:600;">🔑 Enter your API key above</span><br>'
            f'<span style="color:#64748b;font-size:0.72rem;">{meta["hint"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<a href="{meta["link"]}" target="_blank" '
            f'style="color:#60a5fa;font-size:0.8rem;text-decoration:none;font-weight:600;">'
            f'🔗 {meta["link_text"]}</a>',
            unsafe_allow_html=True
        )

    # Groq model selector (only when groq selected)
    if selected_provider == "groq":
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.78rem;margin:10px 0 3px;font-weight:600;">'
            'Model:</p>',
            unsafe_allow_html=True
        )
        selected_groq_model = st.selectbox(
            "Groq Model",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda x: GROQ_MODELS[x],
            label_visibility="collapsed",
            key="groq_model"
        )
    elif selected_provider == "openai":
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.78rem;margin:10px 0 3px;font-weight:600;">'
            'Model:</p>',
            unsafe_allow_html=True
        )
        selected_groq_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            label_visibility="collapsed",
            key="openai_model"
        )
    else:
        selected_groq_model = DEFAULT_GROQ_MODEL

    st.markdown("---")

    # ── HOW TO USE ───────────────────────────────────────────────
    st.markdown(
        '<p style="color:#a78bfa;font-weight:700;font-size:0.85rem;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:8px;">📋 How to Use</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#60a5fa;font-weight:600;font-size:0.82rem;margin:6px 0 4px;">'
        'Tab 1 – Build Your CV</p>',
        unsafe_allow_html=True
    )
    for i, step in enumerate([
        "Pick your AI provider & paste key",
        "Fill in your personal details",
        "Add education & experience",
        "Click ✨ Enhance to AI-optimize",
        "Add skills & generate PDF",
    ], 1):
        st.markdown(
            f'<div class="sidebar-step"><span style="color:#a78bfa;font-weight:700;">{i}.</span>'
            f' <span style="color:#cbd5e1;">{step}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown(
        '<p style="color:#60a5fa;font-weight:600;font-size:0.82rem;margin:10px 0 4px;">'
        'Tab 2 – Analyze CV</p>',
        unsafe_allow_html=True
    )
    for i, step in enumerate([
        "Upload your PDF CV",
        "Paste the Job Description",
        "Get your ATS score instantly",
    ], 1):
        st.markdown(
            f'<div class="sidebar-step"><span style="color:#a78bfa;font-weight:700;">{i}.</span>'
            f' <span style="color:#cbd5e1;">{step}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown(
        '<p style="color:#334155;font-size:0.72rem;">🔒 Keys never stored · session only<br>'
        'Built with Streamlit · Groq · Gemini · OpenAI · FPDF2</p>',
        unsafe_allow_html=True
    )


# ── Helper: get the active runtime key ───────────────────────────
def _get_active_key() -> str:
    """Returns the user's runtime-entered API key for the selected provider."""
    provider = st.session_state.get("ai_provider", "groq")
    sess_map = {
        "groq":   "runtime_groq_key",
        "gemini": "runtime_gemini_key",
        "openai": "runtime_openai_key",
    }
    return st.session_state.get(sess_map.get(provider, "runtime_groq_key"), "").strip()


# ──────────────────────────────────────────────
# Hero Header
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">AI-Powered ATS Resume Builder</h1>', unsafe_allow_html=True)

_provider_label = {
    "groq":   f"⚡ Groq · {GROQ_MODELS.get(st.session_state.get('groq_model', DEFAULT_GROQ_MODEL), '').split('  ')[0]}",
    "gemini": "🔷 Google Gemini",
    "openai": f"🟢 OpenAI · {st.session_state.get('openai_model', 'gpt-4o-mini')}",
}.get(st.session_state.get("ai_provider", "groq"), "⚡ Groq")

st.markdown(
    f'<p class="hero-sub">Build ATS-optimized CVs · Score against any JD'
    f' &nbsp;|&nbsp; <strong style="color:#a78bfa;">{_provider_label}</strong></p>',
    unsafe_allow_html=True
)

# ── Key missing warning banner ────────────────────────────────────
if not _get_active_key():
    st.warning(
        "🔑 **No API key entered.** Paste your key in the sidebar to activate AI features. "
        "Your key stays in this browser session only — never saved anywhere."
    )

# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["✦  Create Professional CV", "◎  Analyze CV Score"])


# ════════════════════════════════════════════════════════════════
# TAB 1 – CV BUILDER
# ════════════════════════════════════════════════════════════════
with tab1:
    # ── Session state init ───────────────────────────────────────
    if "num_exp" not in st.session_state:
        st.session_state.num_exp = 1
    if "num_edu" not in st.session_state:
        st.session_state.num_edu = 1
    if "enhanced_bullets" not in st.session_state:
        st.session_state.enhanced_bullets = {}
    # cv_data stores ALL form values — updated live via callbacks
    if "cv_data" not in st.session_state:
        st.session_state.cv_data = {}

    # ── Callback: fires on every field change, saves value to cv_data ──
    def _save(field):
        """Called on_change — copies widget value into cv_data dict."""
        st.session_state.cv_data[field] = st.session_state.get(field, "")

    # ════════════════════════════════════════════════════════════
    # SECTION 1: Personal Information
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-badge">👤 Step 1 — Personal Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("First Name *",   placeholder="e.g. Ahmad",                key="p_fn",  on_change=_save, args=("p_fn",))
        st.text_input("Email Address *", placeholder="ahmad@example.com",         key="p_em",  on_change=_save, args=("p_em",))
        st.text_input("Location",        placeholder="Islamabad, Pakistan",        key="p_loc", on_change=_save, args=("p_loc",))
    with c2:
        st.text_input("Last Name *",    placeholder="e.g. Khan",                  key="p_ln",  on_change=_save, args=("p_ln",))
        st.text_input("Phone Number",   placeholder="+92 300 0000000",            key="p_ph",  on_change=_save, args=("p_ph",))
        st.text_input("LinkedIn URL",   placeholder="linkedin.com/in/ahmadkhan",  key="p_li",  on_change=_save, args=("p_li",))
    st.text_input("Professional Headline",
                  placeholder="e.g. Senior Software Engineer | Python & AI Specialist",
                  key="p_hl", on_change=_save, args=("p_hl",))
    st.text_area("Professional Summary",
                 placeholder="Write 2-3 sentences about your expertise...",
                 height=100, key="p_sm", on_change=_save, args=("p_sm",))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 2: Education
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-badge">🎓 Step 2 — Education</div>', unsafe_allow_html=True)
    for i in range(st.session_state.num_edu):
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown(f"**Education #{i+1}**")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Degree / Qualification *",  key=f"e_deg_{i}",  placeholder="e.g. BSc Computer Science",       on_change=_save, args=(f"e_deg_{i}",))
            st.text_input("Institution / University *", key=f"e_inst_{i}", placeholder="e.g. COMSATS University Islamabad", on_change=_save, args=(f"e_inst_{i}",))
        with c2:
            st.text_input("Graduation Year",            key=f"e_yr_{i}",   placeholder="e.g. 2023 or 2021–2025",            on_change=_save, args=(f"e_yr_{i}",))
            st.text_input("GPA / Grade (Optional)",     key=f"e_gpa_{i}",  placeholder="e.g. 3.8/4.0",                      on_change=_save, args=(f"e_gpa_{i}",))
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("➕ Add Another Education", key="btn_add_edu"):
        st.session_state.num_edu += 1
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 3: Work Experience
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-badge">💼 Step 3 — Work Experience</div>', unsafe_allow_html=True)
    for i in range(st.session_state.num_exp):
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown(f"**Experience #{i+1}**")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Job Title *",    key=f"x_jt_{i}",  placeholder="e.g. Software Engineer", on_change=_save, args=(f"x_jt_{i}",))
            st.text_input("Company Name *", key=f"x_co_{i}",  placeholder="e.g. Systems Limited",   on_change=_save, args=(f"x_co_{i}",))
            st.text_input("Location",       key=f"x_loc_{i}", placeholder="e.g. Lahore, Pakistan",  on_change=_save, args=(f"x_loc_{i}",))
        with c2:
            st.text_input("Start Date",     key=f"x_sd_{i}",  placeholder="e.g. Jan 2022",          on_change=_save, args=(f"x_sd_{i}",))
            st.text_input("End Date",       key=f"x_ed_{i}",  placeholder="e.g. Present",           on_change=_save, args=(f"x_ed_{i}",))
        st.text_area("Describe Your Responsibilities & Achievements *",
                     key=f"x_raw_{i}", placeholder="e.g. Built REST APIs serving 50K users...",
                     height=120, on_change=_save, args=(f"x_raw_{i}",))

        # ── AI Enhance Bullets ──────────────────────────────────
        if st.button(f"✨ AI Enhance Bullets", key=f"x_enh_{i}"):
            _jt_val  = st.session_state.get(f"x_jt_{i}",  "")
            _raw_val = st.session_state.get(f"x_raw_{i}", "")
            if not _raw_val.strip():
                st.warning("Please describe your experience first.")
            elif not _jt_val.strip():
                st.warning("Please enter a job title.")
            elif not _get_active_key():
                st.error("🔑 Enter your API key in the sidebar first.")
            else:
                with st.spinner("🤖 AI crafting bullet points..."):
                    bullets = enhance_experience_bullets(
                        _raw_val, _jt_val,
                        provider=st.session_state.get("ai_provider", "groq"),
                        api_key=_get_active_key(),
                        groq_model=st.session_state.get("groq_model", DEFAULT_GROQ_MODEL),
                        openai_model=st.session_state.get("openai_model", "gpt-4o-mini"),
                    )
                    st.session_state.enhanced_bullets[i] = bullets

        if i in st.session_state.enhanced_bullets:
            ebs = st.session_state.enhanced_bullets[i]
            if ebs and not ebs[0].startswith("⚠️"):
                st.markdown("**✅ AI-Enhanced Bullet Points:**")
                for b in ebs: st.markdown(f"• {b}")
            elif ebs:
                st.error(ebs[0])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("➕ Add Another Experience", key="btn_add_exp"):
        st.session_state.num_exp += 1
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 4: Skills
    # ════════════════════════════════════════════════════════════
    st.markdown('<div class="section-badge">⚙️ Step 4 — Skills</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown("Enter skills as comma-separated values for each category.")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Technical Skills",  placeholder="Python, SQL, React, Docker",       key="sk_tech", on_change=_save, args=("sk_tech",))
        st.text_input("Soft Skills",        placeholder="Leadership, Communication",        key="sk_soft", on_change=_save, args=("sk_soft",))
    with c2:
        st.text_input("Tools & Platforms",  placeholder="Git, AWS, Jira, VS Code",         key="sk_tool", on_change=_save, args=("sk_tool",))
        st.text_input("Languages (Human)",  placeholder="English (Fluent), Urdu (Native)", key="sk_lang", on_change=_save, args=("sk_lang",))
    st.text_area("Certifications (one per line)",
                 placeholder="AWS Certified Developer – Associate\nGoogle ML Engineer",
                 height=80, key="sk_cert", on_change=_save, args=("sk_cert",))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 5: Projects (Optional)
    # ════════════════════════════════════════════════════════════
    if "num_proj" not in st.session_state:
        st.session_state.num_proj = 0

    st.markdown('<div class="section-badge">🚀 Step 5 — Projects (Optional)</div>', unsafe_allow_html=True)

    if st.session_state.num_proj == 0:
        if st.button("➕ Add Projects Section", key="btn_add_proj_first"):
            st.session_state.num_proj = 1
            st.rerun()
    else:
        for i in range(st.session_state.num_proj):
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown(f"**Project #{i+1}**")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Project Title *",       key=f"pr_title_{i}", placeholder="e.g. MedScan AI", on_change=_save, args=(f"pr_title_{i}",))
                st.text_input("Tech Stack / Tools",    key=f"pr_tech_{i}",  placeholder="e.g. Python, FastAPI, LangChain", on_change=_save, args=(f"pr_tech_{i}",))
            with c2:
                st.text_input("Date / Duration",       key=f"pr_date_{i}",  placeholder="e.g. 2025 or Jan 2025 – Present", on_change=_save, args=(f"pr_date_{i}",))
            st.text_area("Project Description (2-3 key points)",
                         key=f"pr_desc_{i}", placeholder="Describe what you built, tech used, and impact...",
                         height=100, on_change=_save, args=(f"pr_desc_{i}",))
            st.markdown('</div>', unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("➕ Add Another Project", key="btn_add_proj"):
                st.session_state.num_proj += 1
                st.rerun()
        with col_p2:
            if st.session_state.num_proj > 0:
                if st.button("🗑️ Remove Projects Section", key="btn_rm_proj"):
                    st.session_state.num_proj = 0
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # GENERATE BUTTON — reads from session_state (always fresh)
    # ════════════════════════════════════════════════════════════
    if st.button("🚀 Generate My Professional CV", type="primary", key="btn_generate"):

        # Read directly from session_state — guaranteed current values
        _fn  = st.session_state.get("p_fn",  "").strip()
        _ln  = st.session_state.get("p_ln",  "").strip()
        _em  = st.session_state.get("p_em",  "").strip()
        _ph  = st.session_state.get("p_ph",  "").strip()
        _loc = st.session_state.get("p_loc", "").strip()
        _li  = st.session_state.get("p_li",  "").strip()
        _hl  = st.session_state.get("p_hl",  "").strip()
        _sm  = st.session_state.get("p_sm",  "").strip()

        # Build experience list
        _exp_list = []
        for i in range(st.session_state.num_exp):
            _raw = st.session_state.get(f"x_raw_{i}", "")
            _bul = st.session_state.enhanced_bullets.get(i, [])
            if not _bul and _raw.strip():
                _bul = [_raw.strip()]
            _exp_list.append({
                "job_title":  st.session_state.get(f"x_jt_{i}",  ""),
                "company":    st.session_state.get(f"x_co_{i}",  ""),
                "start_date": st.session_state.get(f"x_sd_{i}",  ""),
                "end_date":   st.session_state.get(f"x_ed_{i}",  "") or "Present",
                "location":   st.session_state.get(f"x_loc_{i}", ""),
                "bullets":    _bul,
            })

        # Build education list
        _edu_list = []
        for i in range(st.session_state.num_edu):
            _edu_list.append({
                "degree":      st.session_state.get(f"e_deg_{i}",  ""),
                "institution": st.session_state.get(f"e_inst_{i}", ""),
                "year":        st.session_state.get(f"e_yr_{i}",   ""),
                "gpa":         st.session_state.get(f"e_gpa_{i}",  ""),
            })

        # Build projects list
        _proj_list = []
        for i in range(st.session_state.get("num_proj", 0)):
            _desc = st.session_state.get(f"pr_desc_{i}", "")
            # Split description into bullet lines
            _buls = [l.strip() for l in _desc.split("\n") if l.strip()]
            if not _buls and _desc.strip():
                _buls = [_desc.strip()]
            _proj_list.append({
                "title":   st.session_state.get(f"pr_title_{i}", ""),
                "tech":    st.session_state.get(f"pr_tech_{i}",  ""),
                "date":    st.session_state.get(f"pr_date_{i}",  ""),
                "bullets": _buls,
            })

        # Validate
        errors = []
        if not _fn or not _ln:
            errors.append("Full name (first and last) is required.")
        if not _em:
            errors.append("Email address is required.")
        if not any(e["job_title"].strip() for e in _exp_list):
            errors.append("At least one work experience with a job title is required.")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
        else:
            # Build skills
            _sk_tech = st.session_state.get("sk_tech", "")
            _sk_tool = st.session_state.get("sk_tool", "")
            _sk_soft = st.session_state.get("sk_soft", "")
            _sk_lang = st.session_state.get("sk_lang", "")
            _sk_cert = st.session_state.get("sk_cert", "")
            skills_dict = {}
            if _sk_tech: skills_dict["Technical Skills"]  = [s.strip() for s in _sk_tech.split(",") if s.strip()]
            if _sk_tool: skills_dict["Tools & Platforms"] = [s.strip() for s in _sk_tool.split(",") if s.strip()]
            if _sk_soft: skills_dict["Soft Skills"]       = [s.strip() for s in _sk_soft.split(",") if s.strip()]
            if _sk_lang: skills_dict["Languages"]         = [s.strip() for s in _sk_lang.split(",") if s.strip()]
            certs_list = [c.strip() for c in _sk_cert.split("\n") if c.strip()]

            # AI polish summary
            final_summary = _sm
            primary_jt    = _exp_list[0]["job_title"] if _exp_list else _hl
            full_name     = f"{_fn} {_ln}"
            bad_phrases   = ["i am", "i worked", "i do", "i have", "i was"]
            if _sm and _get_active_key() and (len(_sm.split()) < 30 or any(b in _sm.lower() for b in bad_phrases)):
                with st.spinner("✨ AI is polishing your summary..."):
                    polished = polish_summary(
                        _sm, full_name, primary_jt,
                        provider=st.session_state.get("ai_provider", "groq"),
                        api_key=_get_active_key(),
                        groq_model=st.session_state.get("groq_model", DEFAULT_GROQ_MODEL),
                        openai_model=st.session_state.get("openai_model", "gpt-4o-mini"),
                    )
                    if polished != _sm:
                        final_summary = polished
                        st.info(f"🤖 **AI polished your summary:**\n\n_{final_summary}_")

            cv_payload = {
                "personal": {
                    "first_name": _fn,  "last_name": _ln,
                    "email":      _em,  "phone":     _ph,
                    "location":   _loc, "linkedin":  _li,
                    "headline":   _hl,
                },
                "summary":        final_summary,
                "experiences":    _exp_list,
                "projects":       _proj_list,
                "educations":     _edu_list,
                "skills":         skills_dict,
                "certifications": certs_list,
            }

            with st.spinner("📄 Generating your ATS-optimized PDF..."):
                try:
                    pdf_bytes = generate_cv_pdf(cv_payload)
                    st.session_state.pdf_bytes    = pdf_bytes
                    st.session_state.pdf_filename = f"{_fn}_{_ln}_CV.pdf"
                    st.success("✅ Your CV has been generated successfully!")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Download Your CV (PDF)",
            data=st.session_state.pdf_bytes,
            file_name=st.session_state.get("pdf_filename", "My_CV.pdf"),
            mime="application/pdf",
            key="dl_pdf"
        )
        st.info("💡 **ATS Tip:** Single-column layout — 100% readable by Applicant Tracking Systems.")

    # ════════════════════════════════════════════════════════════
    # ALL FORM FIELDS — collected into plain dicts BEFORE button
    # ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# TAB 2 – CV ANALYZER
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-badge">🔍 CV Analyzer & ATS Scorer</div>', unsafe_allow_html=True)
    st.markdown("Upload your existing CV and paste a job description to get an instant ATS compatibility score with actionable feedback.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown("**📎 Upload Your CV**")
        uploaded_cv = st.file_uploader(
            "Upload PDF CV",
            type=["pdf"],
            help="Your CV must be a PDF file. Scanned images may not extract correctly.",
            label_visibility="collapsed"
        )
        if uploaded_cv:
            st.success(f"✅ Loaded: **{uploaded_cv.name}** ({uploaded_cv.size // 1024} KB)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown("**📋 Paste Job Description**")
        jd_text = st.text_area(
            "Job Description",
            height=160,
            placeholder="Paste the full job description here. Include requirements, responsibilities, and any listed skills/tools...",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Analyze Button ───────────────────────────────────────────
    analyze_clicked = st.button("🔬 Analyze CV vs Job Description", type="primary")

    if analyze_clicked:
        if not uploaded_cv:
            st.error("⚠️ Please upload your CV PDF before analyzing.")
        elif not jd_text.strip():
            st.error("⚠️ Please paste a job description.")
        else:
            # Extract text from PDF
            cv_text = ""
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_cv.read()))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        cv_text += page_text + "\n"
            except Exception as e:
                st.error(f"Failed to read PDF: {str(e)}")
                st.stop()

            if not cv_text.strip():
                st.error("⚠️ Could not extract text from your PDF. It may be a scanned image-based PDF. Please use a text-based PDF.")
                st.stop()

            with st.spinner("🤖 AI is analyzing your CV against the job description..."):
                result = analyze_cv_against_jd(
                    cv_text, jd_text,
                    provider=st.session_state.get("ai_provider", "groq"),
                    api_key=_get_active_key(),
                    groq_model=st.session_state.get("groq_model", DEFAULT_GROQ_MODEL),
                    openai_model=st.session_state.get("openai_model", "gpt-4o-mini"),
                )

            # ── Error Check with detailed diagnostics ────────────
            if "error" in result and result.get("score", 0) == 0:
                err_msg = result['error']
                st.error(f"**Analysis Error:** {err_msg}")
                # Show helpful diagnostics
                with st.expander("🔍 Troubleshooting Help"):
                    active = st.session_state.get("ai_provider", "groq")
                    if active == "groq":
                        st.markdown("""
**You selected: Groq**

1. **Get a free key** at [console.groq.com/keys](https://console.groq.com/keys) — it's instant, no credit card.
2. **Add to `.env`** file (same folder as `app.py`):
   ```
   GROQ_API_KEY=gsk_...your_key_here...
   ```
3. **No quotes** around the key value.
4. **Restart Streamlit** after editing `.env` (`Ctrl+C` → `streamlit run app.py`).
5. **Rate limits:** Groq free tier is generous but if you hit limits, wait 1 minute or switch models.
""")
                    else:
                        st.markdown("""
**You selected: Google Gemini**

1. **Get a free key** at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Add to `.env`** file (same folder as `app.py`):
   ```
   GEMINI_API_KEY=AIza...your_key_here...
   ```
3. **No quotes** around the key value.
4. **Restart Streamlit** after editing `.env`.
""")
                st.stop()

            # ── Score Display ─────────────────────────────────────
            score = result.get("score", 0)
            if score >= 85:
                score_color = "#34d399"
                score_label = "Excellent Match"
                score_emoji = "🟢"
            elif score >= 70:
                score_color = "#60a5fa"
                score_label = "Good Match"
                score_emoji = "🔵"
            elif score >= 50:
                score_color = "#fbbf24"
                score_label = "Moderate Match"
                score_emoji = "🟡"
            else:
                score_color = "#f87171"
                score_label = "Poor Match"
                score_emoji = "🔴"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            # Metrics row
            col_s, col_m, col_miss = st.columns(3)
            with col_s:
                st.metric("ATS Match Score", f"{score}%", delta=score_label)
            with col_m:
                st.metric("✅ Matched Keywords", len(result.get("matched_keywords", [])))
            with col_miss:
                st.metric("❌ Missing Keywords", len(result.get("missing_keywords", [])))

            # Score bar
            st.progress(score / 100)

            # Summary
            if result.get("summary"):
                st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                st.markdown(f"**{score_emoji} Executive Summary**")
                st.markdown(result["summary"])
                st.markdown('</div>', unsafe_allow_html=True)

            # Keywords side by side
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                st.markdown("**✅ Matched Keywords**")
                if result.get("matched_keywords"):
                    pills_html = " ".join([
                        f'<span class="keyword-pill pill-green">{kw}</span>'
                        for kw in result["matched_keywords"]
                    ])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("No strong keyword matches found.")
                st.markdown('</div>', unsafe_allow_html=True)

            with kw_col2:
                st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                st.markdown("**❌ Missing Keywords**")
                if result.get("missing_keywords"):
                    pills_html = " ".join([
                        f'<span class="keyword-pill pill-red">{kw}</span>'
                        for kw in result["missing_keywords"]
                    ])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("No critical keywords missing!")
                st.markdown('</div>', unsafe_allow_html=True)

            # Strengths & Suggestions
            str_col, sug_col = st.columns(2)
            with str_col:
                st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                st.markdown("**💪 Your CV Strengths**")
                for strength in result.get("strengths", []):
                    st.markdown(f"✦ {strength}")
                st.markdown('</div>', unsafe_allow_html=True)

            with sug_col:
                st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                st.markdown("**🚀 Actionable Improvements**")
                for suggestion in result.get("suggestions", []):
                    st.markdown(f"→ {suggestion}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Final advice
            st.markdown("<br>", unsafe_allow_html=True)
            if score < 70:
                st.warning(
                    "💡 **Pro Tip:** Your score is below 70%. Use Tab 1 to rebuild your CV "
                    "with the AI bullet enhancer, incorporating the missing keywords above."
                )
            else:
                st.success(
                    f"🎯 **Great match!** Your CV scores {score}% against this JD. "
                    "Apply with confidence, but consider adding the missing keywords for a higher match."
                )