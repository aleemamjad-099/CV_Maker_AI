"""
pdf_maker.py  —  Professional ATS CV Generator
===============================================
Design inspired by clean modern CV templates:
  - Name left-aligned, large and bold
  - Contact bar with icons (text-based for ATS)
  - Section headers: colored left border accent + full underline
  - Experience: bold role, right-aligned date, italic company
  - Proper bullet indentation with hanging indent
  - Skills: bold category inline with normal text
  - No overflow — all text safely wrapped

FIX: Comprehensive latin-1 sanitization applied at every text output
     point — not just in s() — to prevent font encoding errors with
     Helvetica (italic/bold/regular) which only supports latin-1 range.
"""

import re
import unicodedata
from fpdf import FPDF

# ── Layout constants ──────────────────────────────────────────────
PAGE_W    = 210
PAGE_H    = 297
M_LEFT    = 18
M_RIGHT   = 18
M_TOP     = 16
CONTENT_W = PAGE_W - M_LEFT - M_RIGHT   # 174 mm

# ── Typography ────────────────────────────────────────────────────
LH        = 5.2    # body line height
LH_TIGHT  = 4.8

# ── Colours ───────────────────────────────────────────────────────
BLACK     = (15,  15,  15)
BLUE      = (31,  97, 178)   # accent — section headers
GRAY      = (85,  85,  85)
LGRAY     = (140, 140, 140)
WHITE     = (255, 255, 255)

# ── Master character replacement map ─────────────────────────────
# Covers every common Unicode char that falls outside latin-1 range
# used in Helvetica (built-in PDF font).
_CHAR_MAP = {
    # Quotes & apostrophes
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark
    "\u201a": ",",   # single low-9 quotation mark
    "\u201b": "'",   # single high-reversed-9 quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u201e": '"',   # double low-9 quotation mark
    "\u201f": '"',   # double high-reversed-9 quotation mark

    # Dashes & hyphens
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash  ← THE ORIGINAL CULPRIT
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\u2212": "-",   # minus sign

    # Bullets & markers
    "\u2022": "*",   # bullet
    "\u2023": "*",   # triangular bullet
    "\u2024": ".",   # one dot leader
    "\u2025": "..",  # two dot leader
    "\u2026": "...", # ellipsis
    "\u00b7": "*",   # middle dot
    "\u25cf": "*",   # black circle
    "\u25cb": "*",   # white circle
    "\u25aa": "*",   # black small square
    "\u2043": "-",   # hyphen bullet

    # Spaces
    "\u00a0": " ",   # non-breaking space
    "\u2009": " ",   # thin space
    "\u200a": " ",   # hair space
    "\u2002": " ",   # en space
    "\u2003": " ",   # em space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM

    # Common accented / special chars mapped to ASCII equivalents
    "\u00e0": "a",  "\u00e1": "a",  "\u00e2": "a",  "\u00e3": "a",
    "\u00e4": "a",  "\u00e5": "a",  "\u00e6": "ae",
    "\u00e8": "e",  "\u00e9": "e",  "\u00ea": "e",  "\u00eb": "e",
    "\u00ec": "i",  "\u00ed": "i",  "\u00ee": "i",  "\u00ef": "i",
    "\u00f0": "d",  "\u00f1": "n",  "\u00f2": "o",  "\u00f3": "o",
    "\u00f4": "o",  "\u00f5": "o",  "\u00f6": "o",  "\u00f8": "o",
    "\u00f9": "u",  "\u00fa": "u",  "\u00fb": "u",  "\u00fc": "u",
    "\u00fd": "y",  "\u00ff": "y",  "\u00df": "ss",
    "\u00c0": "A",  "\u00c1": "A",  "\u00c2": "A",  "\u00c3": "A",
    "\u00c4": "A",  "\u00c5": "A",  "\u00c6": "AE",
    "\u00c8": "E",  "\u00c9": "E",  "\u00ca": "E",  "\u00cb": "E",
    "\u00cc": "I",  "\u00cd": "I",  "\u00ce": "I",  "\u00cf": "I",
    "\u00d0": "D",  "\u00d1": "N",  "\u00d2": "O",  "\u00d3": "O",
    "\u00d4": "O",  "\u00d5": "O",  "\u00d6": "O",  "\u00d8": "O",
    "\u00d9": "U",  "\u00da": "U",  "\u00db": "U",  "\u00dc": "U",
    "\u00dd": "Y",
    "\u00e7": "c",  "\u00c7": "C",  # cedilla
    "\u00fe": "th", "\u00de": "TH",  # thorn

    # Latin Extended-A (common in names)
    "\u0100": "A",  "\u0101": "a",  "\u0102": "A",  "\u0103": "a",
    "\u0104": "A",  "\u0105": "a",  "\u0106": "C",  "\u0107": "c",
    "\u010c": "C",  "\u010d": "c",  "\u010e": "D",  "\u010f": "d",
    "\u0110": "D",  "\u0111": "d",  "\u0112": "E",  "\u0113": "e",
    "\u011a": "E",  "\u011b": "e",  "\u011e": "G",  "\u011f": "g",
    "\u0130": "I",  "\u0131": "i",  "\u0141": "L",  "\u0142": "l",
    "\u0143": "N",  "\u0144": "n",  "\u0147": "N",  "\u0148": "n",
    "\u0150": "O",  "\u0151": "o",  "\u0152": "OE", "\u0153": "oe",
    "\u0158": "R",  "\u0159": "r",  "\u015a": "S",  "\u015b": "s",
    "\u015e": "S",  "\u015f": "s",  "\u0160": "S",  "\u0161": "s",
    "\u0162": "T",  "\u0163": "t",  "\u0164": "T",  "\u0165": "t",
    "\u016e": "U",  "\u016f": "u",  "\u0170": "U",  "\u0171": "u",
    "\u0178": "Y",  "\u0179": "Z",  "\u017a": "z",  "\u017b": "Z",
    "\u017c": "z",  "\u017d": "Z",  "\u017e": "z",

    # Misc symbols
    "\u00ae": "(R)", "\u00a9": "(C)", "\u2122": "(TM)",
    "\u00b0": "deg", "\u00b1": "+/-", "\u00d7": "x",
    "\u00f7": "/",   "\u2248": "~",   "\u2260": "!=",
    "\u2264": "<=",  "\u2265": ">=",  "\u221e": "inf",
    "\u2192": "->",  "\u2190": "<-",  "\u2194": "<->",
    "\u25b6": ">",   "\u25c0": "<",

    # Ligatures
    "\ufb00": "ff",  "\ufb01": "fi",  "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
}


def _sanitize(text: str) -> str:
    """
    Two-pass sanitization:
      Pass 1 — explicit map replacements (preserves readability).
      Pass 2 — Unicode NFKD decomposition for anything remaining,
               then strip non-latin-1 bytes as a final safety net.
    The result is guaranteed to encode as latin-1 without error.
    """
    if not text:
        return ""

    # Pass 1: explicit replacements
    for char, replacement in _CHAR_MAP.items():
        text = text.replace(char, replacement)

    # Pass 2: NFKD decompose then drop combining marks
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Pass 3: final hard encode/decode to drop anything still outside latin-1
    text = text.encode("latin-1", errors="replace").decode("latin-1")

    # Clean up replacement characters (the b'?' from errors="replace")
    text = text.replace("?", " ").strip() if text.count("?") > text.count(" ") else text

    return text


class CV(FPDF):

    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(M_LEFT, M_TOP, M_RIGHT)
        self.set_auto_page_break(auto=True, margin=16)
        self.add_page()

    # ── Encoding (public alias — kept for backward compat) ────────
    def s(self, t: str) -> str:
        return _sanitize(t or "")

    def trunc(self, t: str, n: int) -> str:
        t = self.s(t)
        return t if len(t) <= n else t[:n - 2] + ".."

    # ── Name header ───────────────────────────────────────────────
    def name_block(self, first: str, last: str, headline: str):
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*BLACK)
        name = self.s(f"{first} {last}".strip().title())
        self.cell(CONTENT_W, 11, name, ln=True, align="L")

        if headline:
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*BLUE)
            self.cell(CONTENT_W, 6, self.s(headline), ln=True, align="L")

        self.ln(1)
        self._hrule(1.0, BLUE)
        self.ln(3)

    # ── Contact row ───────────────────────────────────────────────
    def contact_block(self, email: str, phone: str, location: str, linkedin: str):
        self.set_font("Helvetica", "", 8.8)
        self.set_text_color(*GRAY)

        parts = []
        if location: parts.append(f"  {self.s(location)}")
        if phone:    parts.append(f"  {self.s(phone)}")
        if email:    parts.append(f"  {self.s(email)}")
        if linkedin:
            url = linkedin.strip()
            if len(url) > 45:
                url = url.replace("https://", "").replace("http://", "").replace("www.", "")
            parts.append(f"  {self.s(url)}")

        line = "   |   ".join(parts)
        self.set_x(M_LEFT)
        self.multi_cell(CONTENT_W, 5.5, line, align="L")
        self.ln(2)

    # ── Section header ────────────────────────────────────────────
    def section(self, title: str):
        self.ln(4)
        x = M_LEFT
        y = self.get_y()
        self.set_fill_color(*BLUE)
        self.rect(x, y, 3, 5.5, style="F")

        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*BLUE)
        self.set_x(M_LEFT + 5)
        self.cell(CONTENT_W - 5, 5.5, self.s(title.upper()), ln=True, align="L")

        self._hrule(0.4, BLUE)
        self.ln(3)

    # ── Horizontal rule ───────────────────────────────────────────
    def _hrule(self, w: float, color: tuple):
        self.set_draw_color(*color)
        self.set_line_width(w)
        y = self.get_y()
        self.line(M_LEFT, y, PAGE_W - M_RIGHT, y)
        self.set_line_width(0.2)

    # ── Summary ───────────────────────────────────────────────────
    def summary_block(self, text: str):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*BLACK)
        self.set_x(M_LEFT)
        self.multi_cell(CONTENT_W, LH + 0.5, self.s(text.strip()), align="J")
        self.ln(1)

    # ── Experience entry ──────────────────────────────────────────
    def experience_entry(self, job_title, company, location, start, end, bullets):
        DATE_W = 36
        BODY_W = CONTENT_W - DATE_W

        # Row 1: Job Title | Date
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLACK)
        self.set_x(M_LEFT)
        self.cell(BODY_W, LH + 1, self.trunc(job_title, 58), ln=False, align="L")

        date_str = ""
        if start or end:
            # Use plain hyphen — never en/em dash
            s = self.s(start or "")
            e = self.s(end or "Present")
            date_str = f"{s} - {e}".strip(" -")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*LGRAY)
        self.cell(DATE_W, LH + 1, date_str, ln=True, align="R")

        # Row 2: Company | Location (italic)
        self.set_x(M_LEFT)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*GRAY)
        comp_str = self.s(company)
        if location:
            comp_str += f"  -  {self.s(location)}"   # plain hyphen, not en-dash
        self.cell(CONTENT_W, LH, comp_str, ln=True, align="L")

        # Bullets
        if bullets:
            self.ln(1)
            for b in bullets:
                b = b.strip().lstrip("*-\u2022\u2013\u2014 ").strip()
                if not b:
                    continue
                self.set_x(M_LEFT + 2)
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(*BLUE)
                self.cell(4, LH, "-", ln=False)
                self.set_font("Helvetica", "", 9.5)
                self.set_text_color(*BLACK)
                self.multi_cell(CONTENT_W - 6, LH, self.s(b), align="L")
        self.ln(2)

    # ── Project entry ─────────────────────────────────────────────
    def project_entry(self, title, tech, date, bullets):
        DATE_W = 36
        BODY_W = CONTENT_W - DATE_W

        self.set_x(M_LEFT)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*BLACK)
        self.cell(BODY_W, LH + 1, self.trunc(title, 60), ln=False)

        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*LGRAY)
        self.cell(DATE_W, LH + 1, self.s(date), ln=True, align="R")

        if tech:
            self.set_x(M_LEFT)
            self.set_font("Helvetica", "I", 8.5)
            self.set_text_color(*GRAY)
            self.multi_cell(CONTENT_W, LH_TIGHT, self.s(tech), align="L")

        for b in bullets:
            b = b.strip().lstrip("*-\u2022\u2013\u2014 ").strip()
            if not b:
                continue
            self.set_x(M_LEFT + 2)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*BLUE)
            self.cell(4, LH, "-", ln=False)
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(*BLACK)
            self.multi_cell(CONTENT_W - 6, LH, self.s(b), align="L")
        self.ln(1.5)

    # ── Education entry ───────────────────────────────────────────
    def education_entry(self, degree, institution, year, gpa):
        DATE_W = 36
        BODY_W = CONTENT_W - DATE_W

        self.set_x(M_LEFT)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLACK)
        self.cell(BODY_W, LH + 1, self.trunc(degree, 60), ln=False)

        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*LGRAY)
        self.cell(DATE_W, LH + 1, self.s(year), ln=True, align="R")

        self.set_x(M_LEFT)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*GRAY)
        inst = self.s(institution)
        if gpa:
            inst += f"  |  GPA: {self.s(gpa)}"
        self.cell(CONTENT_W, LH, inst, ln=True)
        self.ln(2)

    # ── Skills ────────────────────────────────────────────────────
    def skills_block(self, skills_dict: dict):
        self.set_x(M_LEFT)
        for cat, items in skills_dict.items():
            vals = [i.strip() for i in items if i.strip()]
            if not vals:
                continue
            self.set_x(M_LEFT)
            self.set_font("Helvetica", "B", 9.5)
            self.set_text_color(*BLACK)
            self.write(LH, self.s(f"{cat}: "))
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(*GRAY)
            self.write(LH, self.s(", ".join(vals)))
            self.ln(LH)

    # ── Certifications ────────────────────────────────────────────
    def cert_block(self, certs: list):
        for c in certs:
            c = c.strip()
            if not c:
                continue
            self.set_x(M_LEFT + 2)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*BLUE)
            self.cell(4, LH, "-", ln=False)
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(*BLACK)
            self.multi_cell(CONTENT_W - 6, LH, self.s(c), align="L")


# ── Public API ────────────────────────────────────────────────────

def generate_cv_pdf(cv_data: dict) -> bytes:
    pdf = CV()
    p   = cv_data.get("personal", {})

    fn = p.get("first_name", "").strip()
    ln = p.get("last_name",  "").strip()
    hl = p.get("headline",   "").strip()

    # 1. Name + headline
    pdf.name_block(fn, ln, hl)

    # 2. Contact
    pdf.contact_block(
        email    = p.get("email",    ""),
        phone    = p.get("phone",    ""),
        location = p.get("location", ""),
        linkedin = p.get("linkedin", ""),
    )

    # 3. Summary
    summary = cv_data.get("summary", "").strip()
    if summary:
        pdf.section("Professional Summary")
        pdf.summary_block(summary)

    # 4. Skills
    skills = cv_data.get("skills", {})
    has_skills = skills and any(v for v in skills.values() if v)
    if has_skills:
        pdf.section("Technical Skills")
        pdf.skills_block(skills)

    # 5. Experience
    experiences = [e for e in cv_data.get("experiences", []) if e.get("job_title", "").strip()]
    if experiences:
        pdf.section("Professional Experience")
        for exp in experiences:
            raw_bul = exp.get("bullets", [])
            bullets = [b.strip() for b in raw_bul if b.strip() and b.strip() not in ("*", "-", "\u2022", "")]
            pdf.experience_entry(
                job_title = exp.get("job_title", ""),
                company   = exp.get("company",   ""),
                location  = exp.get("location",  ""),
                start     = exp.get("start_date", ""),
                end       = exp.get("end_date",  "Present"),
                bullets   = bullets,
            )

    # 6. Projects
    projects = cv_data.get("projects", [])
    if projects:
        pdf.section("Projects")
        for proj in projects:
            pdf.project_entry(
                title   = proj.get("title",   ""),
                tech    = proj.get("tech",    ""),
                date    = proj.get("date",    ""),
                bullets = proj.get("bullets", []),
            )

    # 7. Education
    educations = [e for e in cv_data.get("educations", []) if e.get("degree", "").strip()]
    if educations:
        pdf.section("Education")
        for edu in educations:
            pdf.education_entry(
                degree      = edu.get("degree",      ""),
                institution = edu.get("institution", ""),
                year        = edu.get("year",        ""),
                gpa         = edu.get("gpa",         ""),
            )

    # 8. Certifications
    certs = [c for c in cv_data.get("certifications", []) if c.strip()]
    if certs:
        pdf.section("Certifications")
        pdf.cert_block(certs)

    return bytes(pdf.output())
