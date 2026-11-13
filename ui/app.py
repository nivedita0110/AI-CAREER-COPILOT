"""
AI Career Copilot — Main Streamlit Application
Production-grade dashboard: Resume ATS Engine, AI Recruiter Feedback,
Skill-Gap Strategist, Resume Rewriter.
"""
import sys
import os

# Ensure the project root is on sys.path regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from app.config import APP_TITLE, APP_SUBTITLE, APP_VERSION, GOOGLE_API_KEY
from app.parser import parse_resume_pdf
from app.scorer import calculate_match
from app.ai_service import (
    generate_recruiter_feedback,
    generate_skill_gap_analysis,
    generate_resume_rewrite,
)
from app.utils import (
    SAMPLE_JOB_DESCRIPTIONS,
    SAMPLE_RESUMES,
    get_score_category,
)
from sentence_transformers import SentenceTransformer


# =============================================================================
# PAGE CONFIG (must be first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM STYLES
# =============================================================================
def inject_styles():
    st.markdown("""
    <style>
    /* ---- Global typography ---- */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }

    /* ---- Accent progress bar ---- */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }

    /* ---- Pills / badge ---- */
    .skill-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .pill-green  { background: #d1fae5; color: #065f46; }
    .pill-red    { background: #fee2e2; color: #991b1b; }
    .pill-yellow { background: #fef3c7; color: #92400e; }
    .pill-blue   { background: #dbeafe; color: #1e40af; }

    /* ---- Score ring label ---- */
    .score-ring {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        line-height: 1;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ---- Section headers ---- */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 4px;
    }

    /* ---- Alert boxes ---- */
    .info-box {
        background: #1e3a5f;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #bfdbfe;
    }

    /* ---- Hide default hamburger / footer ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CACHED RESOURCES
# =============================================================================
@st.cache_resource(show_spinner="Loading embedding model (one-time)…")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =============================================================================
# SESSION STATE BOOTSTRAP
# =============================================================================
def init_session():
    defaults = {
        "page": "home",
        "resume_text": None,
        "resume_meta": None,
        "jd_text": None,
        "ats_result": None,
        "feedback_result": None,
        "gap_result": None,
        "rewrite_result": None,
        "api_key_override": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚀 AI Career Copilot")
        st.markdown(f"<small>v{APP_VERSION}</small>", unsafe_allow_html=True)
        st.divider()

        nav_items = [
            ("🏠", "home",     "Home & Upload"),
            ("📄", "resume",   "Resume Preview"),
            ("📊", "ats",      "ATS Score"),
            ("🧠", "feedback", "AI Feedback"),
            ("📚", "gap",      "Skill Gap Analysis"),
            ("✨", "rewrite",  "Resume Rewriter"),
            ("ℹ️", "about",    "About"),
        ]

        for icon, page, label in nav_items:
            is_active = st.session_state.page == page
            if st.button(f"{icon}  {label}", key=f"nav_{page}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.divider()
        st.markdown('<p class="section-label">API Configuration</p>', unsafe_allow_html=True)
        api_input = st.text_input(
            "Override Gemini API Key",
            value=st.session_state.api_key_override,
            type="password",
            help="Optional. Uses .env GOOGLE_API_KEY if not set here."
        )
        if api_input != st.session_state.api_key_override:
            st.session_state.api_key_override = api_input

        effective_key = st.session_state.api_key_override or GOOGLE_API_KEY
        if effective_key:
            st.success("✅ API key detected")
        else:
            st.error("⚠️ No API key found")

        st.divider()
        st.caption("Disclaimer: This tool assists in resume optimization. It does not guarantee interview selection. AI output should be reviewed before use.")


# =============================================================================
# UTILITY HELPERS
# =============================================================================
def get_api_key():
    return st.session_state.api_key_override or GOOGLE_API_KEY


def render_skill_pills(skills, pill_class="pill-green"):
    if not skills:
        return "<em style='color:#94a3b8;'>None</em>"
    sorted_skills = sorted(list(skills))
    return " ".join(f'<span class="skill-pill {pill_class}">{s}</span>' for s in sorted_skills)


def requires_data(needs_resume=True, needs_jd=True):
    """Renders a warning if the user hasn't uploaded data yet. Returns True if ready."""
    missing = []
    if needs_resume and not st.session_state.resume_text:
        missing.append("resume PDF")
    if needs_jd and not st.session_state.jd_text:
        missing.append("job description")
    if missing:
        st.markdown("""
        <div class='info-box'>
            ⬅️ Please go to <strong>Home &amp; Upload</strong> and provide: """ + " and ".join(missing) + """
        </div>""", unsafe_allow_html=True)
        return False
    return True


# =============================================================================
# PAGES
# =============================================================================

# ----- HOME PAGE -----
def page_home():
    st.markdown(f"# 🚀 {APP_TITLE}")
    st.markdown(f"#### {APP_SUBTITLE}")
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-label">📄 Resume Upload</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload your resume (PDF, max 10 MB)",
            type=["pdf"],
            help="Scanned-only PDFs cannot be parsed. Use a text-based PDF for best results.",
            label_visibility="collapsed"
        )
        if uploaded:
            with st.spinner("Extracting resume text…"):
                meta = parse_resume_pdf(uploaded)
            if not meta["success"]:
                st.error(f"❌ {meta['error']}")
            elif meta["is_scanned"]:
                st.warning(
                    f"⚠️ Possible scanned/image PDF detected ({meta['word_count']} words extracted). "
                    "OCR is not available. Please upload a text-based PDF."
                )
            else:
                st.session_state.resume_text = meta["text"]
                st.session_state.resume_meta = meta
                # Clear cached results when new resume is uploaded
                st.session_state.ats_result = None
                st.session_state.feedback_result = None
                st.session_state.gap_result = None
                st.session_state.rewrite_result = None
                st.success(
                    f"✅ Resume parsed: **{meta['page_count']}** page(s), "
                    f"**{meta['word_count']:,}** words."
                )

        # Sample Resume Quick-load
        st.markdown('<p class="section-label" style="margin-top:16px">Or load a sample resume</p>', unsafe_allow_html=True)
        sample_key = st.selectbox("Sample Resumes", ["— select —"] + list(SAMPLE_RESUMES.keys()), label_visibility="collapsed")
        if sample_key != "— select —":
            if st.button("Load Sample Resume", key="load_sample_resume"):
                text = SAMPLE_RESUMES[sample_key]
                st.session_state.resume_text = text
                st.session_state.resume_meta = {
                    "page_count": 1,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "is_scanned": False,
                    "error": None
                }
                st.session_state.ats_result = None
                st.session_state.feedback_result = None
                st.session_state.gap_result = None
                st.session_state.rewrite_result = None
                st.success(f"✅ Loaded: {sample_key}")

    with col_right:
        st.markdown('<p class="section-label">📌 Job Description</p>', unsafe_allow_html=True)
        jd_input = st.text_area(
            "Paste the full job description here",
            value=st.session_state.jd_text or "",
            height=240,
            placeholder="Paste the job description here…",
            label_visibility="collapsed"
        )

        # Sample JD Quick-load
        st.markdown('<p class="section-label" style="margin-top:4px">Or load a sample JD</p>', unsafe_allow_html=True)
        jd_key = st.selectbox("Sample JDs", ["— select —"] + list(SAMPLE_JOB_DESCRIPTIONS.keys()), label_visibility="collapsed")
        if jd_key != "— select —":
            if st.button("Load Sample JD", key="load_sample_jd"):
                st.session_state.jd_text = SAMPLE_JOB_DESCRIPTIONS[jd_key]
                st.session_state.ats_result = None
                st.session_state.feedback_result = None
                st.session_state.gap_result = None
                st.session_state.rewrite_result = None
                st.rerun()

    if jd_input and jd_input.strip() != (st.session_state.jd_text or "").strip():
        st.session_state.jd_text = jd_input.strip()
        st.session_state.ats_result = None
        st.session_state.feedback_result = None
        st.session_state.gap_result = None
        st.session_state.rewrite_result = None

    st.markdown("---")
    # Status banner
    has_resume = bool(st.session_state.resume_text)
    has_jd = bool(st.session_state.jd_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Resume", "✅ Loaded" if has_resume else "❌ Not loaded")
    c2.metric("Job Description", "✅ Loaded" if has_jd else "❌ Not loaded")
    c3.metric("ATS Score", "✅ Computed" if st.session_state.ats_result else "⏳ Pending")

    if has_resume and has_jd:
        st.markdown("")
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("⚡ Run ATS Analysis", type="primary", use_container_width=True):
                embedder = load_embedder()
                with st.spinner("Computing ATS score and skill match…"):
                    result = calculate_match(
                        embedder=embedder,
                        resume=st.session_state.resume_text,
                        jd=st.session_state.jd_text
                    )
                st.session_state.ats_result = result
                st.session_state.page = "ats"
                st.rerun()


# ----- RESUME PREVIEW PAGE -----
def page_resume_preview():
    st.markdown("## 📄 Resume Preview")
    if not requires_data(needs_resume=True, needs_jd=False):
        return

    meta = st.session_state.resume_meta or {}
    if meta:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages", meta.get("page_count", "—"))
        m2.metric("Words", f"{meta.get('word_count', 0):,}")
        m3.metric("Characters", f"{meta.get('char_count', 0):,}")

    st.markdown("---")
    st.text_area("Extracted Text", st.session_state.resume_text, height=550, label_visibility="collapsed")


# ----- ATS SCORE PAGE -----
def page_ats_score():
    st.markdown("## 📊 ATS Score Dashboard")
    if not requires_data():
        return

    # Compute / fetch cached result
    if not st.session_state.ats_result:
        embedder = load_embedder()
        with st.spinner("Computing ATS analysis…"):
            result = calculate_match(
                embedder=embedder,
                resume=st.session_state.resume_text,
                jd=st.session_state.jd_text
            )
        st.session_state.ats_result = result
    else:
        result = st.session_state.ats_result

    score = result["final_score"]
    category = get_score_category(score)

    # Score headline
    c_score, c_meta = st.columns([1, 2], gap="large")
    with c_score:
        st.markdown(f'<div class="score-ring">{score:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:6px;">'
            f'<span style="background:{category["color"]};color:#fff;padding:4px 14px;border-radius:12px;font-weight:700;font-size:0.85rem;">'
            f'{category["label"]}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<p style="text-align:center;color:#94a3b8;font-size:0.82rem;margin-top:8px;">{category["message"]}</p>', unsafe_allow_html=True)
        st.progress(score / 100)

    with c_meta:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Semantic Match", f"{result['semantic_score']:.1f}%")
        m2.metric("Skill Coverage", f"{result['skill_score']:.1f}%")
        m3.metric("Matched Skills", len(result["matched_skills"]))
        m4.metric("Missing Skills", len(result["missing_skills"]))

        # Experience comparison
        exp_resume = result.get("resume_experience", [])
        exp_jd = result.get("jd_experience", [])
        if exp_jd:
            st.markdown(f"**JD requires:** {', '.join(exp_jd)}")
        if exp_resume:
            st.markdown(f"**Resume states:** {', '.join(exp_resume)}")

    st.markdown("---")

    # Skill breakdowns
    tab_matched, tab_partial, tab_missing, tab_audit, tab_detail = st.tabs([
        "✅ Matched Skills", "🟡 Partial Match", "❌ Missing Skills",
        "🗂 Resume Sections", "🔬 Detailed Skill View"
    ])

    with tab_matched:
        if result["matched_skills"]:
            st.markdown(render_skill_pills(result["matched_skills"], "pill-green"), unsafe_allow_html=True)
        else:
            st.info("No firmly matched skills detected. Consider tailoring keyword usage.")

    with tab_partial:
        if result.get("partial_skills"):
            st.markdown(render_skill_pills(result["partial_skills"], "pill-yellow"), unsafe_allow_html=True)
            st.caption("These skills are closely related or adjacent to JD requirements but do not exactly match.")
        else:
            st.info("No partially matched skills.")

    with tab_missing:
        if result["missing_skills"]:
            st.markdown(render_skill_pills(result["missing_skills"], "pill-red"), unsafe_allow_html=True)
            st.caption("These skills were expected in the JD but not found in your resume text.")
        else:
            st.success("No missing critical skills detected! 🎉")

    with tab_audit:
        st.markdown("### Resume Section Completeness")
        sections = result.get("section_audit", {})
        if sections:
            for section, present in sections.items():
                icon = "✅" if present else "❌"
                st.markdown(f"{icon} **{section}**")
        else:
            st.info("Section audit not available.")

    with tab_detail:
        st.markdown("### Skill-by-Skill Match Breakdown")
        details = result.get("skill_details", [])
        if details:
            status_map = {"matched": "✅", "partial": "🟡", "missing": "❌"}
            for item in details[:60]:  # Cap display at 60 rows
                icon = status_map.get(item["status"], "—")
                sim = item["similarity"]
                matched_with = item.get("matched_with", "None")
                st.markdown(
                    f"{icon} **{item['jd_skill']}** "
                    f"→ `{matched_with}` ({sim:.0f}% sim)"
                )
        else:
            st.info("No detail data available.")

    if st.button("⚡ Recompute ATS Score", key="recompute"):
        st.session_state.ats_result = None
        st.rerun()


# ----- AI FEEDBACK PAGE -----
def page_ai_feedback():
    st.markdown("## 🧠 AI Recruiter Feedback")
    if not requires_data():
        return

    if not st.session_state.ats_result:
        st.info("Running ATS analysis first…")
        embedder = load_embedder()
        with st.spinner("Computing ATS analysis…"):
            st.session_state.ats_result = calculate_match(
                embedder=load_embedder(),
                resume=st.session_state.resume_text,
                jd=st.session_state.jd_text
            )

    result = st.session_state.ats_result
    score = result["final_score"]
    category = get_score_category(score)

    st.markdown(
        f'<div class="info-box">ATS Score: <strong>{score:.1f}%</strong> — '
        f'<span style="color:#93c5fd">{category["label"]}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Use cached feedback, or generate
    if st.session_state.feedback_result:
        fb = st.session_state.feedback_result
        st.markdown(fb["content"])
        st.caption(f"Generated using model: `{fb.get('model_used', 'unknown')}`")
        col_regen, _ = st.columns([1, 3])
        with col_regen:
            if st.button("🔄 Regenerate Feedback", key="regen_feedback"):
                st.session_state.feedback_result = None
                st.rerun()
    else:
        if st.button("🧠 Generate Recruiter Feedback", type="primary", key="gen_feedback"):
            api_key = get_api_key()
            with st.spinner("Consulting AI recruiter…"):
                fb = generate_recruiter_feedback(
                    resume=st.session_state.resume_text,
                    jd=st.session_state.jd_text,
                    ats_score=score,
                    matched_skills=result["matched_skills"],
                    missing_skills=result["missing_skills"],
                    partial_skills=result.get("partial_skills", set()),
                    api_key=api_key
                )
            if fb["success"]:
                st.session_state.feedback_result = fb
                st.rerun()
            else:
                st.error(f"❌ {fb['error']}")


# ----- SKILL GAP PAGE -----
def page_skill_gap():
    st.markdown("## 📚 Skill Gap Analysis & Upskilling Roadmap")
    if not requires_data():
        return

    if not st.session_state.ats_result:
        embedder = load_embedder()
        with st.spinner("Computing ATS analysis…"):
            st.session_state.ats_result = calculate_match(
                embedder=embedder,
                resume=st.session_state.resume_text,
                jd=st.session_state.jd_text
            )

    result = st.session_state.ats_result
    missing = result["missing_skills"]
    partial = result.get("partial_skills", set())

    if not missing and not partial:
        st.success("🎉 No significant skill gaps detected! Your resume covers the JD well.")
        return

    missing_pills = render_skill_pills(missing, "pill-red")
    partial_pills = render_skill_pills(partial, "pill-yellow")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label">Missing Skills</p>', unsafe_allow_html=True)
        st.markdown(missing_pills, unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="section-label">Partial / Adjacent Skills</p>', unsafe_allow_html=True)
        st.markdown(partial_pills, unsafe_allow_html=True)

    st.markdown("")

    if st.session_state.gap_result:
        gap = st.session_state.gap_result
        st.markdown(gap["content"])
        st.caption(f"Generated using model: `{gap.get('model_used', 'unknown')}`")
        col_regen, _ = st.columns([1, 3])
        with col_regen:
            if st.button("🔄 Regenerate Analysis", key="regen_gap"):
                st.session_state.gap_result = None
                st.rerun()
    else:
        if st.button("📚 Generate Skill Gap Roadmap", type="primary", key="gen_gap"):
            api_key = get_api_key()
            with st.spinner("Building upskilling roadmap…"):
                gap = generate_skill_gap_analysis(
                    jd=st.session_state.jd_text,
                    missing_skills=missing,
                    partial_skills=partial,
                    api_key=api_key
                )
            if gap["success"]:
                st.session_state.gap_result = gap
                st.rerun()
            else:
                st.error(f"❌ {gap['error']}")


# ----- RESUME REWRITER PAGE -----
def page_resume_rewriter():
    st.markdown("## ✨ AI Resume Rewriter")
    st.markdown(
        '<div class="info-box">ℹ️ The rewriter <strong>will not</strong> fabricate skills, jobs, or achievements. '
        'It strengthens your existing content using ATS-optimized language and structure.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    if not requires_data():
        return

    if not st.session_state.ats_result:
        embedder = load_embedder()
        with st.spinner("Computing ATS analysis…"):
            st.session_state.ats_result = calculate_match(
                embedder=embedder,
                resume=st.session_state.resume_text,
                jd=st.session_state.jd_text
            )

    result = st.session_state.ats_result

    if st.session_state.rewrite_result:
        rw = st.session_state.rewrite_result
        st.markdown("### 📄 ATS-Optimized Resume")
        st.markdown(rw["content"])
        st.caption(f"Generated using model: `{rw.get('model_used', 'unknown')}`")
        st.download_button(
            label="⬇ Download as Markdown",
            data=rw["content"],
            file_name="ats_optimized_resume.md",
            mime="text/markdown"
        )
        col_regen, _ = st.columns([1, 3])
        with col_regen:
            if st.button("🔄 Regenerate Resume", key="regen_rewrite"):
                st.session_state.rewrite_result = None
                st.rerun()
    else:
        if st.button("✨ Generate Optimized Resume", type="primary", key="gen_rewrite"):
            api_key = get_api_key()
            with st.spinner("Rewriting and optimizing resume — this may take 15-30 seconds…"):
                rw = generate_resume_rewrite(
                    resume=st.session_state.resume_text,
                    jd=st.session_state.jd_text,
                    matched_skills=result["matched_skills"],
                    missing_skills=result["missing_skills"],
                    api_key=api_key
                )
            if rw["success"]:
                st.session_state.rewrite_result = rw
                st.rerun()
            else:
                st.error(f"❌ {rw['error']}")


# ----- ABOUT PAGE -----
def page_about():
    st.markdown("## ℹ️ About AI Career Copilot")
    st.markdown(f"""
---
**Version:** {APP_VERSION}

### What This App Does
AI Career Copilot helps software engineering candidates maximize their chances of passing Applicant Tracking Systems (ATS) by:
- Parsing resume PDFs and analyzing them against job descriptions.
- Computing a composite ATS match score using semantic embeddings + skill keyword analysis.
- Providing detailed AI-generated recruiter feedback, skill-gap roadmaps, and ATS-optimized resume rewrites.

### Technology Stack
| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit |
| Embedding Model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| AI Engine | Google Gemini (via LangChain) |
| PDF Parsing | PyMuPDF |
| Similarity Engine | Scikit-learn + NumPy |
| NLP | Regex-based + Semantic Matching |

### How ATS Scoring Works
The score is a **weighted composite**:
- **70%** — Semantic document similarity (chunk-averaged embedding cosine similarity)
- **30%** — Skill coverage (vectorized batch matching against JD-extracted terms)

> **Note:** ATS score is an *estimate* based on keyword and semantic alignment. Real ATS systems vary significantly by company and platform.

### Ethical Disclaimer
- The AI rewriter **never fabricates** experience, degrees, or achievements.
- It reframes your *existing* content using stronger, ATS-aligned language.
- Claiming skills you do not have is dishonest and may lead to disqualification.

### Limitations
- Scanned / image-only PDFs cannot be parsed (requires OCR which is not included).
- ATS scoring does not perfectly simulate any specific vendor's ATS system.
- AI quality depends on Gemini API availability and quota.
---
""")


# =============================================================================
# MAIN ROUTER
# =============================================================================
def main():
    inject_styles()
    init_session()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        page_home()
    elif page == "resume":
        page_resume_preview()
    elif page == "ats":
        page_ats_score()
    elif page == "feedback":
        page_ai_feedback()
    elif page == "gap":
        page_skill_gap()
    elif page == "rewrite":
        page_resume_rewriter()
    elif page == "about":
        page_about()
    else:
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()
