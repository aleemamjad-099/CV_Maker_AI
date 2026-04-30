# ⚡ ATS CV Pro – AI-Powered Resume Builder & Analyzer

A production-grade Streamlit application that uses **Google Gemini 1.5 Flash** to:
1. **Build** ATS-optimized CVs with AI-enhanced bullet points → export as PDF
2. **Analyze** any existing CV against a job description → get an ATS match score

---

## 📁 Project Structure

```
ats_cv_builder/
│
├── app.py                  # Main Streamlit UI (two tabs)
│
├── core/
│   ├── __init__.py
│   ├── ai_engine.py        # Gemini AI: bullet enhancement + CV analysis
│   └── pdf_maker.py        # FPDF2: ATS-safe single-column PDF generation
│
├── .env                    # Your API keys (create this, not committed to git)
├── .env.example            # Template for .env
├── requirements.txt        # All Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Setup

### 1. Clone / Download the project
```bash
cd ats_cv_builder
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Then edit .env and add your key:
# GEMINI_API_KEY=your_actual_key_here
```
Get a **free** Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🔧 Tech Stack

| Layer        | Technology                         |
|-------------|-------------------------------------|
| Frontend    | Streamlit 1.35                      |
| AI / LLM    | Google Gemini 1.5 Flash             |
| PDF Output  | FPDF2 (ATS-safe, text-based PDF)    |
| PDF Parsing | PyPDF2                              |
| Env Config  | python-dotenv                       |

---

## 📄 ATS Compatibility Notes

The generated PDF follows strict ATS rules:
- ✅ Single-column layout (no tables/columns)
- ✅ Standard Helvetica font (no decorative fonts)
- ✅ Plain text content (no images or icons in body)
- ✅ Logical reading order
- ✅ No headers/footers with critical info

---

## 🛡️ Error Handling

- Empty form fields are validated before PDF generation
- AI API errors return graceful fallbacks
- Invalid PDF uploads show clear error messages
- JSON parsing from AI responses is fault-tolerant