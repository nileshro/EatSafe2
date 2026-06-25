import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from PIL import Image

from database.db import (
    init_db, save_user, get_user, get_profile_dict,
    save_scan, get_scan_history, get_dashboard_stats, get_all_scores
)
from database.auth_db import create_auth_table, register_user, login_user
from services.gemini_service import analyze_product
from services.scoring_service import calculate_health_score
from services.pdf_service import generate_report
from services.risk_analysis import analyze_risks
from services.ingredient_analyzer import analyze_ingredients
from services.recommendation_service import recommend_alternatives
from services.logo_service import predict_logo
from services.authenticity_service import calculate_authenticity

# ── INIT ──────────────────────────────────────────────────────────────────────
init_db()
create_auth_table()

st.set_page_config(
    page_title="EatSafe – Know What's In Your Food",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "EatSafe – AI-powered food product verification & nutrition advisor."
    }
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <meta name="description"
          content="EatSafe – AI-powered food product verification and nutrition advisor.">
    <meta name="keywords"
          content="food safety, nutrition analyzer, health score, AI food checker, EatSafe">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta property="og:title"       content="EatSafe – Know What's In Your Food">
    <meta property="og:description" content="AI-powered food product verification and nutrition advisor">
    <meta property="og:type"        content="website">

    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div  { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background   : rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        padding      : 10px 14px !important;
        margin       : 3px 0 !important;
        border       : 1px solid transparent !important;
        font-size    : 14px !important;
        font-weight  : 500 !important;
        cursor       : pointer !important;
        color        : #e2e8f0 !important;
        display      : flex !important;
        align-items  : center !important;
        transition   : all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background  : rgba(99,102,241,0.18) !important;
        border-color: rgba(99,102,241,0.35) !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background  : rgba(239,68,68,0.12) !important;
        color       : #f87171 !important;
        border      : 1px solid rgba(239,68,68,0.3) !important;
        border-radius: 10px !important;
        width       : 100% !important;
        font-weight : 600 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background  : rgba(239,68,68,0.22) !important;
        border-color: #ef4444 !important;
        transform   : none !important;
    }

    /* ── Main layout ─────────────────────────────────── */
    .main .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1100px; }

    h2 { font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:700 !important;
         font-size:1.5rem !important; color:#1e293b !important; }
    h3 { font-size:1rem !important; font-weight:600 !important;
         color:#374151 !important; margin-top:1.2rem !important; }

    /* ── Cards ───────────────────────────────────────── */
    .es-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 24px; margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.2s;
    }
    .es-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); }

    .feat-card {
        background: linear-gradient(135deg,#f8faff,#f0f4ff);
        border: 1px solid #e0e7ff; border-radius: 16px;
        padding: 24px 16px; text-align: center; height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(99,102,241,0.12); }
    .feat-icon  { font-size: 2.4rem; margin-bottom: 10px; }
    .feat-title { font-weight: 700; font-size: 0.92rem; color: #1e293b; margin-bottom: 4px; }
    .feat-desc  { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

    /* ── Hero banner ─────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 55%,#a78bfa 100%);
        border-radius: 20px; padding: 48px 40px; color: white;
        margin-bottom: 20px; position: relative; overflow: hidden;
    }
    .hero-banner::before {
        content: ''; position: absolute; top: -50%; right: -8%;
        width: 380px; height: 380px; background: rgba(255,255,255,0.07);
        border-radius: 50%; pointer-events: none;
    }
    .hero-title {
        font-family: 'Plus Jakarta Sans',sans-serif;
        font-size: 2.1rem; font-weight: 800; line-height: 1.2;
        margin-bottom: 12px; position: relative;
    }
    .hero-sub {
        font-size: 0.98rem; opacity: 0.9;
        margin-bottom: 20px; line-height: 1.65; position: relative;
    }
    .hero-chips { display: flex; gap: 8px; flex-wrap: wrap; position: relative; }
    .hero-chip  {
        background: rgba(255,255,255,0.16); border-radius: 50px;
        padding: 5px 14px; font-size: 0.78rem; font-weight: 600;
    }

    /* ── Chips ───────────────────────────────────────── */
    .stat-chip {
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.22);
        border-radius: 50px; padding: 4px 13px;
        font-size: 0.78rem; font-weight: 600; color: #6366f1;
        display: inline-block; margin: 3px 3px 0 0;
    }

    /* ── Scan History rows ───────────────────────────── */
    .history-row {
        display: flex; align-items: center; justify-content: space-between;
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 14px 20px; margin-bottom: 10px; transition: background 0.15s, border-color 0.15s;
    }
    .history-row:hover { background: #f0f4ff; border-color: #c7d2fe; }
    .history-product { font-weight: 600; color: #1e293b; font-size: 0.93rem; }
    .history-brand   { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    .history-score   { font-family: 'Plus Jakarta Sans',sans-serif; font-weight: 800; font-size: 1.3rem; }
    .history-date    { font-size: 0.73rem; color: #94a3b8; text-align: right; margin-top: 2px; }

    /* ── Step bar ────────────────────────────────────── */
    .step-bar  {
        display: flex; gap: 0; margin-bottom: 20px;
        background: #f1f5f9; border-radius: 12px; padding: 4px;
    }
    .step-item {
        flex: 1; text-align: center; padding: 10px 8px; border-radius: 10px;
        font-size: 0.79rem; font-weight: 600; color: #94a3b8; transition: all 0.2s;
    }
    .step-item.active { background: white; color: #6366f1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .step-item.done   { color: #16a34a; }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 0.9rem !important; transition: opacity 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.92 !important; transform: translateY(-1px) !important; }

    /* ── Progress ────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg,#6366f1,#8b5cf6) !important; border-radius: 4px !important;
    }

    /* ── Metrics ─────────────────────────────────────── */
    [data-testid="metric-container"] {
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:16px !important;
    }
    [data-testid="metric-container"] label {
        color:#64748b !important; font-size:0.76rem !important;
        font-weight:600 !important; text-transform:uppercase; letter-spacing:0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family:'Plus Jakarta Sans',sans-serif !important;
        font-size:1.8rem !important; font-weight:800 !important; color:#1e293b !important;
    }

    /* ── Misc ────────────────────────────────────────── */
    hr { border:none !important; border-top:1px solid #f1f5f9 !important; margin:20px 0 !important; }
    [data-testid="stFileUploader"] {
        border:2px dashed #c7d2fe !important; border-radius:14px !important;
        background:#f8faff !important; padding:12px !important;
    }

    /* ── Dashboard ───────────────────────────────────── */
    .dash-stat {
        background: linear-gradient(135deg,#6366f1,#8b5cf6);
        border-radius:16px; padding:24px; color:white; text-align:center;
    }
    .dash-stat-num   { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.8rem; font-weight:800; line-height:1; }
    .dash-stat-label { font-size:0.78rem; opacity:0.85; margin-top:5px; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; }

    .dash-best  {
        background: linear-gradient(135deg,#f0fdf4,#dcfce7); border:1px solid #86efac;
        border-radius:14px; padding:18px 20px; display:flex; justify-content:space-between;
        align-items:center; margin-bottom:10px;
    }
    .dash-worst {
        background: linear-gradient(135deg,#fef2f2,#fee2e2); border:1px solid #fca5a5;
        border-radius:14px; padding:18px 20px; display:flex; justify-content:space-between; align-items:center;
    }

    /* ── Auth ────────────────────────────────────────── */
    .auth-wrap {
        max-width:440px; margin:40px auto; background:white;
        border:1px solid #e2e8f0; border-radius:20px; padding:40px;
        box-shadow:0 4px 24px rgba(0,0,0,0.07);
    }

    /* ── Mobile ──────────────────────────────────────── */
    @media (max-width:768px) {
        .main .block-container { padding:1rem 1rem 2rem !important; }
        .hero-title { font-size:1.4rem; }
        .hero-banner { padding:24px 20px; }
        .hero-banner::before { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "user"        not in st.session_state: st.session_state.user        = None
if "scan_step"   not in st.session_state: st.session_state.scan_step   = 1
if "scan_images" not in st.session_state: st.session_state.scan_images = {}
if "nav_target"    not in st.session_state: st.session_state.nav_target    = None
if "scan_result"   not in st.session_state: st.session_state.scan_result   = None
if "scan_auth"     not in st.session_state: st.session_state.scan_auth     = None
if "scan_score"    not in st.session_state: st.session_state.scan_score    = None
if "scan_risk"     not in st.session_state: st.session_state.scan_risk     = None
if "scan_warnings" not in st.session_state: st.session_state.scan_warnings = None
if "scan_alts"     not in st.session_state: st.session_state.scan_alts     = None

# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.user is None:

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:24px;">
            <div style="font-size:2.5rem;">🍎</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;
                        font-size:1.7rem;color:#1e293b;">EatSafe</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:4px;">
                Know what's really inside your food</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        em = st.text_input("Email",    placeholder="you@example.com", key="li_em")
        pw = st.text_input("Password", type="password", placeholder="Your password", key="li_pw")
        if st.button("Sign In →", key="login_btn"):
            if not em or not pw:
                st.error("Please fill in all fields.")
            else:
                user = login_user(em, pw)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with tab_reg:
        nm  = st.text_input("Full Name",         placeholder="Your name",        key="rg_nm")
        em2 = st.text_input("Email",             placeholder="you@example.com",  key="rg_em")
        pw2 = st.text_input("Password",          type="password", placeholder="Min 6 characters", key="rg_pw")
        pw3 = st.text_input("Confirm Password",  type="password", placeholder="Repeat password",  key="rg_cf")
        if st.button("Create Account →", key="reg_btn"):
            if not nm or not em2 or not pw2:
                st.error("All fields are required.")
            elif pw2 != pw3:
                st.error("Passwords do not match.")
            elif len(pw2) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                if register_user(nm, em2, pw2):
                    st.success("Account created! Please sign in.")
                else:
                    st.error("Email already registered.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="padding:16px 8px 18px;">
            <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;">🍎 EatSafe</div>
            <div style="font-size:0.73rem;color:#64748b;margin-top:2px;">AI Food Analyzer</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1.2px;'
        'color:#475569;font-weight:700;padding:0 4px;margin-bottom:6px;">Navigation</div>',
        unsafe_allow_html=True
    )

    _nav_options = ["🏠  Home", "👤  Profile", "📷  Scan Product", "📜  Scan History", "📊  Dashboard"]
    _nav_index   = (
        _nav_options.index(st.session_state.nav_target)
        if st.session_state.nav_target in _nav_options
        else 0
    )
    # Reset nav_target so it only fires once
    st.session_state.nav_target = None

    page = st.radio(
        "Navigation",
        _nav_options,
        index=_nav_index,
        label_visibility="collapsed",
        key="main_menu"
    )

    uname  = st.session_state.user["name"]
    uemail = st.session_state.user["email"]
    st.markdown(
        f"""
        <div style="margin-top:16px;background:rgba(99,102,241,0.13);
                    border:1px solid rgba(99,102,241,0.28);border-radius:12px;
                    padding:12px 14px;margin-bottom:10px;">
            <div style="font-size:0.72rem;color:#818cf8;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.5px;">Signed in as</div>
            <div style="font-size:0.88rem;color:#f1f5f9;font-weight:700;margin-top:3px;">{uname}</div>
            <div style="font-size:0.71rem;color:#64748b;margin-top:1px;">{uemail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Sign Out", key="logout_btn"):
        st.session_state.user        = None
        st.session_state.scan_step   = 1
        st.session_state.scan_images = {}
        st.rerun()

email = st.session_state.user["email"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Know What's Really<br>Inside Your Food 🔍</div>
            <div class="hero-sub">
                AI-powered nutrition analysis, ingredient warnings,<br>
                health scoring and brand authenticity — all in seconds.
            </div>
            <div class="hero-chips">
                <span class="hero-chip">✅ Gemini AI</span>
                <span class="hero-chip">🛡️ Brand Verification</span>
                <span class="hero-chip">📊 Health Scoring</span>
                <span class="hero-chip">🌿 Smart Alternatives</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("📷  Start Scanning Now →", key="hero_scan"):
        st.session_state.nav_target = "📷  Scan Product"
        st.rerun()

    st.markdown("---")
    st.markdown("### What EatSafe Does For You")

    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, desc) in zip(
        [c1, c2, c3, c4],
        [
            ("📷", "Scan Product",  "Camera or upload — front pack, nutrition label & ingredients"),
            ("🤖", "AI Analysis",   "Gemini AI extracts full nutrition facts and ingredient list"),
            ("💯", "Health Score",  "Personalised 1–10 score based on your health profile"),
            ("🛡️", "Authenticity", "CNN model verifies the product brand is genuine"),
        ]
    ):
        with col:
            st.markdown(
                f'<div class="feat-card">'
                f'<div class="feat-icon">{icon}</div>'
                f'<div class="feat-title">{title}</div>'
                f'<div class="feat-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### How It Works")

    s1, s2, s3 = st.columns(3)
    for col, (num, title, desc) in zip(
        [s1, s2, s3],
        [
            ("1️⃣", "Upload 3 Images", "Front of pack, nutrition label, and ingredients list"),
            ("2️⃣", "AI Processes",    "Gemini converts all values to per-100 g standard automatically"),
            ("3️⃣", "Get Your Report", "Score, risks, ingredient warnings, alternatives & PDF export"),
        ]
    ):
        with col:
            st.markdown(
                f'<div class="es-card" style="text-align:center;">'
                f'<div style="font-size:1.9rem;margin-bottom:10px;">{num}</div>'
                f'<div style="font-weight:700;font-size:0.9rem;color:#1e293b;margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;line-height:1.55;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Profile":

    st.markdown("## Health Profile")
    st.markdown(
        "<p style='color:#64748b;margin-top:-6px;'>Personalise your score by telling us about your conditions.</p>",
        unsafe_allow_html=True
    )

    col_form, col_saved = st.columns(2, gap="large")

    with col_form:
        st.markdown('<div class="es-card">', unsafe_allow_html=True)
        st.markdown("**Update Your Conditions**")
        diabetes     = st.checkbox("🩺 Diabetes",                     help="Flags products with high sugar")
        hypertension = st.checkbox("❤️ Hypertension",                 help="Flags products with high sodium")
        obesity      = st.checkbox("⚖️ Obesity / Weight Management",  help="Flags high-calorie products")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Save Profile", key="save_profile"):
            save_user(email, int(diabetes), int(hypertension), int(obesity))
            st.success("Profile saved successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_saved:
        ud = get_user(email)
        if ud:
            st.markdown('<div class="es-card">', unsafe_allow_html=True)
            st.markdown("**Saved Profile**")

            def _badge(val, label):
                colour = "#16a34a" if val else "#94a3b8"
                bg     = "#f0fdf4" if val else "#f8fafc"
                text   = "Active"  if val else "Not set"
                return (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:0.87rem;color:#374151;">{label}</span>'
                    f'<span style="background:{bg};color:{colour};border-radius:20px;'
                    f'padding:3px 12px;font-size:0.75rem;font-weight:600;">{text}</span></div>'
                )

            st.markdown(
                _badge(ud[1], "🩺 Diabetes") +
                _badge(ud[2], "❤️ Hypertension") +
                _badge(ud[3], "⚖️ Obesity"),
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No profile saved yet. Fill in the form and save.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCAN PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📷  Scan Product":

    st.markdown("## Scan Product")
    st.markdown(
        "<p style='color:#64748b;margin-top:-6px;'>"
        "Upload or capture 3 images: front package, nutrition label, and ingredients.</p>",
        unsafe_allow_html=True
    )

    capture_mode = st.radio(
        "Input method",
        ["📷  Camera", "🖼️  Upload Images"],
        horizontal=True,
        label_visibility="collapsed",
        key="capture_mode"
    )

    images_ready = False
    images       = []

    # ── CAMERA ────────────────────────────────────────────────────────────────
    if capture_mode == "📷  Camera":
        st.markdown('<div class="es-card">', unsafe_allow_html=True)

        step        = st.session_state.scan_step
        step_labels = ["Front Package", "Nutrition Label", "Ingredients List"]
        step_hints  = [
            "Capture the full front/back of the product clearly",
            "Ensure every nutrition value is legible",
            "Make sure all ingredients text is clearly visible"
        ]

        steps_html = ""
        for i, lbl in enumerate(["1 · Front", "2 · Nutrition", "3 · Ingredients"], 1):
            cls = "active" if i == step else ("done" if i < step else "")
            chk = "✓ " if i < step else ""
            steps_html += f'<div class="step-item {cls}">{chk}{lbl}</div>'
        st.markdown(f'<div class="step-bar">{steps_html}</div>', unsafe_allow_html=True)

        st.markdown(f"**Step {step} of 3 — {step_labels[step - 1]}**")
        st.caption(step_hints[step - 1])

        cam = st.camera_input(f"📸 Capture: {step_labels[step - 1]}", key=f"cam_{step}")

        if cam:
            st.session_state.scan_images[step] = Image.open(cam)
            if step < 3:
                if st.button(f"Next → {step_labels[step]}", key="next_step_btn"):
                    st.session_state.scan_step = step + 1
                    st.rerun()
            else:
                if st.button("✅  Analyse Product", key="cam_analyze"):
                    if len(st.session_state.scan_images) == 3:
                        images       = [st.session_state.scan_images[i] for i in [1, 2, 3]]
                        images_ready = True
                    else:
                        st.warning("Please capture all 3 images first.")

        if st.session_state.scan_images:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            thumbs = st.columns(3)
            for i, (idx, img) in enumerate(st.session_state.scan_images.items()):
                with thumbs[i]:
                    st.image(img, caption=f"✓ {step_labels[idx - 1]}", width='stretch')

        _, rst_col = st.columns([4, 1])
        with rst_col:
            if st.button("🔄 Reset", key="reset_btn"):
                st.session_state.scan_step   = 1
                st.session_state.scan_images = {}
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── UPLOAD ────────────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="es-card">', unsafe_allow_html=True)
        st.markdown("**Upload 3 images** in order: front package → nutrition label → ingredients")
        uploaded = st.file_uploader(
            "Drop images here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded:
            if len(uploaded) < 3:
                st.warning(f"Need at least 3 images. ({len(uploaded)}/3 uploaded)")
            else:
                images       = [Image.open(f) for f in uploaded]
                images_ready = True
                prev_cols    = st.columns(3)
                labels_up    = ["Front Package", "Nutrition Label", "Ingredients"]
                for i, (col, f) in enumerate(zip(prev_cols, uploaded[:3])):
                    with col:
                        st.image(f, caption=labels_up[i], width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ANALYSE ───────────────────────────────────────────────────────────────
    if images_ready:
        if st.button("🔬  Analyse Product", key="analyze_btn"):
            with st.spinner("Analysing with Gemini AI… (10–20 seconds)"):
                try:
                    logo_result = (
                        predict_logo(images[0]) if images
                        else {"brand": "Unknown", "confidence": 0}
                    )
                    result = analyze_product(images)

                    # Handle quota exceeded
                    if result.get("_quota_exceeded"):
                        st.error(
                            "⏳ **Gemini API quota exceeded** (free tier: 20 requests/day). "
                            "Please wait a few minutes and try again, or check your API billing at "
                            "https://ai.dev/rate-limit"
                        )
                        st.stop()

                    if result["confidence"] < 0.5:
                        st.error(
                            "⚠️ Could not reliably extract nutrition data. "
                            "Please ensure the nutrition label is clearly visible and well-lit."
                        )
                        st.stop()

                    auth_score   = calculate_authenticity(
                        result.get("brand"), logo_result["brand"], logo_result["confidence"]
                    )
                    profile      = get_profile_dict(email)
                    score        = calculate_health_score(result["nutrition_per_100g"], profile)

                    save_scan(email, result.get("product_name"), result.get("brand"), score)

                    risk_result  = analyze_risks(result["nutrition_per_100g"], profile)
                    ing_warnings = analyze_ingredients(result["ingredients"])
                    alternatives = recommend_alternatives(
                        result.get("product_name"), result["nutrition_per_100g"]
                    )

                    # Store results in session state so PDF button survives reruns
                    st.session_state.scan_result   = result
                    st.session_state.scan_auth     = auth_score
                    st.session_state.scan_score    = score
                    st.session_state.scan_risk     = risk_result
                    st.session_state.scan_warnings = ing_warnings
                    st.session_state.scan_alts     = alternatives

                    # ── Results ───────────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")

                    r1, r2 = st.columns([1, 2], gap="large")

                    with r1:
                        sc  = "#16a34a" if score >= 7 else "#d97706" if score >= 4 else "#dc2626"
                        sbg = "#f0fdf4" if score >= 7 else "#fffbeb" if score >= 4 else "#fef2f2"
                        sbd = "#86efac" if score >= 7 else "#fde68a" if score >= 4 else "#fca5a5"
                        st.markdown(
                            f'<div style="background:{sbg};border:2px solid {sbd};border-radius:20px;'
                            f'padding:28px;text-align:center;margin-bottom:12px;">'
                            f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:4rem;'
                            f'font-weight:800;color:{sc};line-height:1;">{score}</div>'
                            f'<div style="font-size:0.8rem;color:{sc};font-weight:600;margin-top:6px;">'
                            f'Health Score / 10</div></div>',
                            unsafe_allow_html=True
                        )
                        st.progress(score / 10)

                    with r2:
                        # Handle "Possibly X" prefix for brand matching
                        _cnn_raw   = logo_result.get("raw_class", logo_result["brand"]).lower()
                        _gem_brand = (result.get("brand") or "").lower()
                        brand_match = (
                            _cnn_raw == _gem_brand or
                            _gem_brand in _cnn_raw or
                            _cnn_raw in _gem_brand
                        )
                        st.markdown(
                            f'<div class="es-card">'
                            f'<div style="font-size:1.2rem;font-weight:800;color:#1e293b;">'
                            f'{result.get("product_name", "—")}</div>'
                            f'<div style="color:#64748b;font-size:0.83rem;margin-top:4px;">'
                            f'Brand: <strong>{result.get("brand", "Unknown")}</strong></div>'
                            f'<hr style="margin:12px 0 10px;">'
                            f'<span class="stat-chip">🤖 AI Confidence: {result.get("confidence", 0)*100:.0f}%</span>'
                            f'<span class="stat-chip">🛡️ Authenticity: {auth_score:.0f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        if logo_result["brand"] == "Unknown":
                            st.info("ℹ️ Logo model could not identify brand — manual verification recommended.")
                        elif brand_match:
                            st.success(
                                f"✅ Brand Verified: {logo_result['brand']} "
                                f"(CNN confidence: {logo_result['confidence']*100:.1f}%)"
                            )
                        else:
                            st.warning(
                                f"⚠️ CNN detected **{logo_result['brand']}** "
                                f"— label says **{result.get('brand', '')}**. May still be genuine."
                            )

                    st.markdown("---")

                    # Nutrition grid
                    st.markdown("### 🥗 Nutrition Facts *(per 100 g)*")
                    nutrition = result["nutrition_per_100g"]
                    n_cols    = st.columns(6)
                    for col, (icon, label, val) in zip(n_cols, [
                        ("⚡", "Energy",  f"{nutrition.get('energy',  0)} kcal"),
                        ("💪", "Protein", f"{nutrition.get('protein', 0)} g"),
                        ("🫒", "Fat",     f"{nutrition.get('fat',     0)} g"),
                        ("🍬", "Sugar",   f"{nutrition.get('sugar',   0)} g"),
                        ("🧂", "Sodium",  f"{nutrition.get('sodium',  0)} mg"),
                        ("🌾", "Fiber",   f"{nutrition.get('fiber',   0)} g"),
                    ]):
                        with col:
                            st.markdown(
                                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                f'border-radius:14px;padding:14px;text-align:center;">'
                                f'<div style="font-size:1.25rem;">{icon}</div>'
                                f'<div style="font-size:0.97rem;font-weight:800;color:#1e293b;margin:4px 0;">{val}</div>'
                                f'<div style="font-size:0.68rem;color:#64748b;font-weight:600;text-transform:uppercase;">{label}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                    # Bar chart
                    chart_df = pd.DataFrame({
                        "Nutrient": ["Protein", "Fat", "Sugar", "Fiber", "Sat. Fat"],
                        "Value": [
                            nutrition.get("protein",       0),
                            nutrition.get("fat",           0),
                            nutrition.get("sugar",         0),
                            nutrition.get("fiber",         0),
                            nutrition.get("saturated_fat", 0)
                        ]
                    })
                    fig = px.bar(
                        chart_df, x="Nutrient", y="Value",
                        color="Nutrient",
                        color_discrete_sequence=["#6366f1","#f59e0b","#ef4444","#10b981","#8b5cf6"],
                        title="Nutrition Breakdown (per 100 g)",
                        labels={"Value": "Amount (g)"}
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter", title_font_size=14,
                        margin=dict(t=40, b=20, l=0, r=0),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(gridcolor="#f1f5f9")
                    )
                    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y} g<extra></extra>")
                    st.plotly_chart(fig, width='stretch')

                    st.markdown("---")

                    # Risk / Suitable / Not Suitable
                    ra, rb, rc = st.columns(3)
                    with ra:
                        st.markdown("### ⚠️ Risk Factors")
                        if risk_result["risks"]:
                            for r in risk_result["risks"]:
                                st.warning(r)
                        else:
                            st.success("No major risks detected")

                    with rb:
                        st.markdown("### ✅ Suitable For")
                        if risk_result["suitable_for"]:
                            for item in risk_result["suitable_for"]:
                                st.success(item)
                        else:
                            st.info("—")

                    with rc:
                        st.markdown("### 🚫 Not Suitable For")
                        if risk_result["not_suitable_for"]:
                            for item in risk_result["not_suitable_for"]:
                                st.error(item)
                        else:
                            st.success("No restrictions for your profile")

                    st.markdown("---")

                    # Ingredient warnings + alternatives
                    ia, ib = st.columns(2, gap="large")
                    with ia:
                        st.markdown("### 🧪 Ingredient Warnings")
                        if ing_warnings:
                            for w in ing_warnings:
                                st.warning(w)
                        else:
                            st.success("No concerning ingredients found")

                    with ib:
                        st.markdown("### 🌿 Healthier Alternatives")
                        if alternatives:
                            for a in alternatives:
                                st.success(a)
                        else:
                            st.info("No specific alternatives available")

                    st.markdown("---")

                    # PDF export — generate immediately and store bytes
                    st.markdown("### 📄 Export Report")
                    _pdf_path = generate_report(
                        "EatSafe_Report.pdf", result, score, auth_score,
                        risk_result, ing_warnings, alternatives
                    )
                    with open(_pdf_path, "rb") as _f:
                        st.session_state.scan_pdf_bytes = _f.read()

                    st.session_state.scan_step   = 1
                    st.session_state.scan_images = {}

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # ── Persistent results from previous analysis ───────────────────────────
    if st.session_state.scan_result is not None:
        st.markdown("---")
        st.markdown("### 📄 Export Last Report")
        if "scan_pdf_bytes" in st.session_state and st.session_state.scan_pdf_bytes:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state.scan_pdf_bytes,
                file_name="EatSafe_Report.pdf",
                mime="application/pdf",
                key="pdf_download_persistent"
            )
        else:
            # Generate on demand if bytes not cached
            if st.button("📄 Generate & Download PDF", key="pdf_gen_btn"):
                _r  = st.session_state.scan_result
                _s  = st.session_state.scan_score
                _a  = st.session_state.scan_auth
                _rk = st.session_state.scan_risk
                _w  = st.session_state.scan_warnings
                _al = st.session_state.scan_alts
                _path = generate_report("EatSafe_Report.pdf", _r, _s, _a, _rk, _w, _al)
                with open(_path, "rb") as _f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=_f.read(),
                        file_name="EatSafe_Report.pdf",
                        mime="application/pdf",
                        key="pdf_download_gen"
                    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCAN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜  Scan History":

    st.markdown("## Scan History")
    st.markdown(
        "<p style='color:#64748b;margin-top:-6px;'>All products you've analysed with EatSafe.</p>",
        unsafe_allow_html=True
    )

    history = get_scan_history(email)

    if not history:
        st.markdown(
            '<div class="es-card" style="text-align:center;padding:52px;">'
            '<div style="font-size:3rem;margin-bottom:14px;">📭</div>'
            '<div style="font-weight:700;color:#1e293b;font-size:1.05rem;">No scans yet</div>'
            '<div style="color:#64748b;font-size:0.84rem;margin-top:6px;">'
            'Go to Scan Product to analyse your first item.</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='color:#94a3b8;font-size:0.82rem;margin-bottom:14px;'>"
            f"{len(history)} product(s) scanned</div>",
            unsafe_allow_html=True
        )
        for product, brand, score, scan_time in history:
            sc = "#16a34a" if score >= 7 else "#d97706" if score >= 4 else "#dc2626"
            try:
                dt    = datetime.strptime(scan_time, "%Y-%m-%d %H:%M:%S")
                d_str = dt.strftime("%d %b %Y")
                t_str = dt.strftime("%I:%M %p")
            except Exception:
                d_str, t_str = scan_time, ""

            st.markdown(
                f'<div class="history-row">'
                f'<div>'
                f'<div class="history-product">{product or "Unknown Product"}</div>'
                f'<div class="history-brand">{brand or "Unknown Brand"}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div class="history-score" style="color:{sc};">{score}'
                f'<span style="font-size:0.82rem;color:#94a3b8;font-weight:500;">/10</span></div>'
                f'<div class="history-date">{d_str}<br>{t_str}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":

    st.markdown("## Dashboard")
    st.markdown(
        "<p style='color:#64748b;margin-top:-6px;'>Your food scanning overview and health trends.</p>",
        unsafe_allow_html=True
    )

    stats   = get_dashboard_stats(email)
    history = get_all_scores(email)

    avg       = round(stats["avg_score"], 1) if stats["avg_score"] else 0
    low_count = sum(1 for _, s, _ in history if s < 5) if history else 0

    # 3 stat cards
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            f'<div class="dash-stat">'
            f'<div class="dash-stat-num">{stats["total_scans"]}</div>'
            f'<div class="dash-stat-label">Total Scans</div></div>',
            unsafe_allow_html=True
        )
    with d2:
        st.markdown(
            f'<div class="dash-stat" style="background:linear-gradient(135deg,#059669,#10b981);">'
            f'<div class="dash-stat-num">{avg}</div>'
            f'<div class="dash-stat-label">Avg Health Score</div></div>',
            unsafe_allow_html=True
        )
    with d3:
        st.markdown(
            f'<div class="dash-stat" style="background:linear-gradient(135deg,#dc2626,#ef4444);">'
            f'<div class="dash-stat-num">{low_count}</div>'
            f'<div class="dash-stat-label">Low-Score Products</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Best / Worst
    bw1, bw2 = st.columns(2)
    with bw1:
        if stats["best_product"]:
            bp, bs = stats["best_product"]
            st.markdown(
                f'<div class="dash-best">'
                f'<div>'
                f'<div style="font-size:0.7rem;font-weight:700;color:#166534;'
                f'text-transform:uppercase;letter-spacing:0.5px;">🏆 Best Product</div>'
                f'<div style="font-weight:700;color:#14532d;font-size:0.92rem;margin-top:4px;">{bp}</div>'
                f'</div>'
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:2rem;'
                f'font-weight:800;color:#16a34a;">{bs}'
                f'<span style="font-size:0.95rem;color:#86efac;">/10</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
    with bw2:
        if stats["worst_product"]:
            wp, ws = stats["worst_product"]
            st.markdown(
                f'<div class="dash-worst">'
                f'<div>'
                f'<div style="font-size:0.7rem;font-weight:700;color:#991b1b;'
                f'text-transform:uppercase;letter-spacing:0.5px;">⚠️ Needs Attention</div>'
                f'<div style="font-weight:700;color:#7f1d1d;font-size:0.92rem;margin-top:4px;">{wp}</div>'
                f'</div>'
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:2rem;'
                f'font-weight:800;color:#dc2626;">{ws}'
                f'<span style="font-size:0.95rem;color:#fca5a5;">/10</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    if history:
        df = pd.DataFrame(history, columns=["Product", "Score", "Date"])

        # Line chart
        st.markdown("### 📈 Health Score Trend")
        fig_line = px.line(
            df, x="Date", y="Score", markers=True,
            hover_data={"Product": True, "Score": True, "Date": False}
        )
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", margin=dict(t=10, b=20, l=0, r=0),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(gridcolor="#f1f5f9", range=[0, 11], title="Score")
        )
        fig_line.update_traces(
            line_color="#6366f1", line_width=3,
            marker=dict(size=10, color="#6366f1", line=dict(width=2, color="white")),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{y}/10<extra></extra>"
        )
        st.plotly_chart(fig_line, width='stretch')

        # Bar chart
        st.markdown("### 📊 Score by Product")
        fig_bar = px.bar(
            df, x="Product", y="Score",
            color="Score",
            color_continuous_scale=[[0, "#dc2626"], [0.5, "#d97706"], [1, "#16a34a"]],
            range_color=[0, 10],
            text="Score"
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", margin=dict(t=10, b=20, l=0, r=0),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(gridcolor="#f1f5f9", range=[0, 11], title="Score"),
            coloraxis_showscale=False
        )
        fig_bar.update_traces(
            texttemplate="%{text}/10",
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Score: %{y}/10<extra></extra>"
        )
        st.plotly_chart(fig_bar, width='stretch')

        # Full table
        st.markdown("### 📋 All Scans")
        st.dataframe(
            df.rename(columns={"Product": "Product", "Score": "Health Score", "Date": "Scanned On"}),
            width='stretch',
            hide_index=True
        )

    else:
        st.markdown(
            '<div class="es-card" style="text-align:center;padding:52px;">'
            '<div style="font-size:3rem;margin-bottom:14px;">📊</div>'
            '<div style="font-weight:700;color:#1e293b;">No data yet</div>'
            '<div style="color:#64748b;font-size:0.84rem;margin-top:6px;">'
            'Scan your first product to populate your dashboard.</div></div>',
            unsafe_allow_html=True
        )