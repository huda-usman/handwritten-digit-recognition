"""
Digit Recognition Studio — 2-Page Streamlit App
Run : streamlit run app.py
Deps: pip install streamlit streamlit-drawable-canvas tensorflow opencv-python pillow numpy
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="DIGITAI — Digit Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

# ── Session defaults ────────────────────────────────────────────────────────
for k, v in {
    "page":           "home",
    "canvas_key":     0,
    "up_result":      None,
    "draw_result":    None,
    "active_src":     None,
    "show_all":       False,
    "canvas_imgdata": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main > div {
  font-family: 'Inter', sans-serif !important;
  background-color: #EEF2FF !important;
  color: #1E1B4B !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
.stMarkdown { margin-bottom: 0 !important; }
div[data-testid="column"] { padding: 0 6px !important; }


/* ─── NAV ─── */
.topnav {
  background: #fff; 
  border-bottom: 2px solid #E0E7FF;
  padding: 0 72px; 
  height: 76px;
  display: flex; 
  align-items: center; 
  justify-content: space-between;
  box-shadow: 0 4px 20px rgba(79,70,229,.08);
}

.nav-brand { 
  display: flex; 
  align-items: center; 
  gap: 12px; 
  font-size: 20px;
  font-weight: 800; 
  color: #1E1B4B; 
  letter-spacing: -.02em;
}

.nav-icon {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(135deg,#4F46E5,#7C3AED);
  display: flex; 
  align-items: center; 
  justify-content: center;
  font-size: 20px;
  color: #fff; 
  box-shadow: 0 6px 14px rgba(79,70,229,.35);
}

.nav-links { 
  display: flex; 
  align-items: center;
  gap: 40px;
}

.nav-links a {
  text-decoration: none;
  color: inherit;
}

.navlnk, .nav-links a {
  font-size: 15px;
  font-weight: 600;
  color: #4B5563;
  padding: 8px 0;
  position: relative;
  transition: color .2s ease;
  cursor: pointer;
  display: inline-block;
}

.navlnk:hover, .nav-links a:hover { 
  color: #4F46E5; 
}

.navlnk-act, .nav-links a.active { 
  color: #4F46E5 !important; 
  font-weight: 700;
}

.navlnk-act::after, .nav-links a.active::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg,#4F46E5,#818CF8);
  border-radius: 3px 3px 0 0;
}

/* ─── HOME ─── */
.home-page {
  height: calc(100vh - 60px);
  background: linear-gradient(145deg,#EEF2FF 0%,#EDE9FE 35%,#E0E7FF 65%,#EEF2FF 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 80px;
  position: relative;
  overflow: hidden;
  gap: 260px;
}

.horb1 {
  position: absolute; top: -80px; right: 25%;
  width: 500px; height: 500px; border-radius: 50%;
  background: radial-gradient(circle,rgba(99,102,241,.2) 0%,transparent 70%);
  pointer-events: none;
}
.horb2 {
  position: absolute; bottom: -60px; left: 2%;
  width: 400px; height: 400px; border-radius: 50%;
  background: radial-gradient(circle,rgba(124,58,237,.15) 0%,transparent 70%);
  pointer-events: none;
}

/* ── LEFT ── */
.home-left {
  flex: 0 1 auto;
  max-width: 500px;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.hpill {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,.9); border: 1px solid rgba(79,70,229,.25);
  border-radius: 100px; padding: 8px 18px;
  font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: #4F46E5; margin-bottom: 20px; backdrop-filter: blur(8px);
  box-shadow: 0 2px 8px rgba(79,70,229,.1);
}

.hh1 {
  font-size: 56px; font-weight: 900; line-height: 1.0;
  color: #1E1B4B; letter-spacing: -.03em; margin: 0;
}

.hh1g {
  font-size: 56px; font-weight: 900; line-height: 1.05;
  background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 55%,#A78BFA 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; letter-spacing: -.03em; margin: 0 0 16px;
}

.hsub {
  font-size: 16px; color: #4B5563; line-height: 1.6;
  max-width: 420px; margin: 0 0 24px;
  font-weight: 400;
}

/* CTA buttons */
.hcta { 
  display: flex; 
  gap: 16px; 
  margin-bottom: 28px; 
}

.hbtn-primary {
  display: inline-flex; align-items: center; gap: 8px;
  background: linear-gradient(135deg,#4F46E5,#7C3AED);
  color: #fff !important; font-size: 15px; font-weight: 700;
  padding: 12px 26px; border-radius: 12px; border: none;
  box-shadow: 0 8px 20px rgba(79,70,229,.4);
  text-decoration: none !important; cursor: pointer;
  transition: all .2s ease;
}

.hbtn-primary:hover { 
  box-shadow: 0 12px 28px rgba(79,70,229,.5); 
  transform: translateY(-2px); 
}

.hbtn-secondary {
  display: inline-flex; align-items: center; gap: 8px;
  background: #fff; border: 2px solid #C7D2FE;
  color: #4F46E5 !important; font-size: 15px; font-weight: 600;
  padding: 11px 24px; border-radius: 12px;
  text-decoration: none !important; cursor: pointer;
  transition: all .2s ease;
}

.hbtn-secondary:hover { 
  background: #EEF2FF; 
  border-color: #4F46E5; 
  transform: translateY(-2px);
}

.hstats {
  display: flex; 
  gap: 36px; 
  padding-top: 20px;
  border-top: 2px solid rgba(79,70,229,.15);
}

.hstat-item {
  display: flex;
  flex-direction: column;
}

.sv { 
  font-size: 24px; 
  font-weight: 800; 
  color: #1E1B4B; 
  letter-spacing: -.02em; 
  line-height: 1.2;
}

.sl { 
  font-size: 11px; 
  font-weight: 600; 
  color: #6B7280; 
  letter-spacing: .1em; 
  text-transform: uppercase; 
  margin-top: 4px; 
}

/* ── RIGHT — cards ── */
.home-right {
  display: flex;
  flex-direction: column;
  gap: 16px;
  z-index: 1;
  width: 300px;
  flex-shrink: 0;
}

.fcard {
  background: rgba(255,255,255,.95); 
  border: 1px solid rgba(224,231,255,.9);
  border-radius: 20px; 
  padding: 20px 22px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 24px rgba(79,70,229,.1);
  transition: all .25s ease;
}

.fcard:hover { 
  transform: translateY(-4px); 
  box-shadow: 0 16px 32px rgba(79,70,229,.18);
  border-color: rgba(79,70,229,.3);
}

.fcard-inner { 
  display: flex; 
  align-items: flex-start; 
  gap: 16px; 
}

.ficon { 
  width: 44px; 
  height: 44px; 
  border-radius: 14px; 
  display: flex; 
  align-items: center; 
  justify-content: center; 
  font-size: 22px; 
  flex-shrink: 0; 
  box-shadow: 0 4px 12px rgba(0,0,0,.05);
}

.fi-a { 
  background: linear-gradient(135deg,#EEF2FF,#C7D2FE);
  color: #4F46E5;
}

.fi-b { 
  background: linear-gradient(135deg,#F0FDF4,#BBF7D0);
  color: #16A34A;
}

.fi-c { 
  background: linear-gradient(135deg,#FFF7ED,#FED7AA);
  color: #EA580C;
}

.ft { 
  font-size: 15px; 
  font-weight: 800; 
  color: #1E1B4B; 
  margin-bottom: 6px; 
  letter-spacing: -.02em;
}

.fd { 
  font-size: 13px; 
  color: #6B7280; 
  line-height: 1.5; 
  font-weight: 400;
}

/* ─── BUTTONS ─── */
.stButton > button {
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  border-radius: 10px !important; border: none !important;
  transition: all .18s !important; cursor: pointer !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg,#4F46E5,#7C3AED) !important;
  color: #fff !important; box-shadow: 0 4px 14px rgba(79,70,229,.35) !important;
  padding: 11px 24px !important; font-size: 14px !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 8px 22px rgba(79,70,229,.45) !important; transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  background: #fff !important; color: #374151 !important;
  border: 1.5px solid #E5E7EB !important;
  padding: 10px 22px !important; font-size: 14px !important;
}
.stButton > button[kind="secondary"]:hover { border-color: #4F46E5 !important; color: #4F46E5 !important; }

/* ── Kill scrollbar: zero out Streamlit wrappers ── */
[data-testid="stMain"] > div:first-child,
[data-testid="stVerticalBlock"] { gap: 0 !important; }
button[data-testid="baseButton-secondary"][key="clr"],
.clear-btn button {
  background: #FFF1F2 !important; color: #E11D48 !important;
  border-color: #FECDD3 !important; padding: 7px 14px !important; font-size: 12px !important;
}

/* ─── STUDIO ──────────────────────────────────────────────────────────── */
.studio-hdr {
  background: linear-gradient(135deg, #F8FAFF, #EDE9FE);
  border-bottom: 1px solid #E0E7FF;
  padding: 24px 52px 20px;
  box-shadow: 0 2px 12px rgba(79,70,229,0.05);
}

.studio-title { 
  font-size: 26px; 
  font-weight: 800; 
  color: #1E1B4B; 
  letter-spacing: -0.02em; 
  margin-bottom: 4px;
}

.studio-sub { 
  font-size: 14px; 
  color: #6B7280; 
  font-weight: 400;
}

/* ─── MAIN LAYOUT ─────────────────────────────────────────────────────── */
.studio-main {
  padding: 24px 52px;
  background: #F5F7FF;
  min-height: calc(100vh - 140px);
}

/* Make columns equal height */
[data-testid="column"] {
  height: fit-content;
}

/* ─── PANEL ───────────────────────────────────────────────────────────── */
.panel {
  background: #fff;
  border: 1px solid #E0E7FF;
  border-radius: 20px;
  box-shadow: 0 8px 20px rgba(79,70,229,0.06);
  overflow: hidden;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
  max-height: 600px;
}

.panel:hover {
  box-shadow: 0 12px 28px rgba(79,70,229,0.1);
}

.ph { 
  padding: 16px 20px; 
  border-bottom: 1px solid #F3F4F6; 
  display: flex; 
  align-items: center; 
  justify-content: space-between;
  background: #FCFDFF;
  flex-shrink: 0;
}

.ph-l { 
  display: flex; 
  align-items: center; 
  gap: 10px; 
}

.phi { 
  width: 32px; 
  height: 32px; 
  border-radius: 10px; 
  display: flex; 
  align-items: center; 
  justify-content: center; 
  font-size: 15px; 
  font-weight: 600;
}

.phi-up { 
  background: linear-gradient(135deg, #EEF2FF, #C7D2FE);
  color: #4F46E5;
}

.phi-cnv { 
  background: linear-gradient(135deg, #F0FDF4, #BBF7D0);
  color: #16A34A;
}

.phi-res { 
  background: linear-gradient(135deg, #FFF7ED, #FED7AA);
  color: #EA580C;
}

.ph-title { 
  font-size: 12px; 
  font-weight: 700; 
  letter-spacing: 0.05em; 
  text-transform: uppercase; 
  color: #374151; 
}

.pb { 
  padding: 20px;
  overflow-y: auto;
  flex: 1;
  max-height: 500px;
}

/* ─── UPLOAD SECTION ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  margin-bottom: 16px !important;
}

[data-testid="stFileUploader"] > div:first-child {
  border: none !important; 
  background: transparent !important; 
  padding: 0 !important; 
  box-shadow: none !important;
}

[data-testid="stFileUploader"] label { 
  display: none !important; 
}

[data-testid="stFileUploader"] > div > div {
  border: 2px dashed #C7D2FE !important; 
  border-radius: 16px !important;
  background: #FFFFFF !important; 
  padding: 32px 20px !important;
  text-align: center !important;
  transition: all 0.2s ease !important;
  margin-bottom: 20px !important;
}

[data-testid="stFileUploader"] > div > div:hover { 
  border-color: #4F46E5 !important; 
  background: #F5F7FF !important;
}

[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
  color: white !important;
  border: none !important;
  border-radius: 30px !important;
  padding: 10px 24px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  margin-top: 16px !important;
  box-shadow: 0 4px 10px rgba(79,70,229,0.3) !important;
  transition: all 0.2s ease !important;
  display: inline-block !important;
  width: auto !important;
}

[data-testid="stFileUploader"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 16px rgba(79,70,229,0.4) !important;
}

/* File info text - GUARANTEED VISIBLE */
[data-testid="stFileUploader"] [data-testid="stMarkdown"] {
  color: #1E1B4B !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  margin-top: 12px !important;
  padding: 12px 16px !important;
  background: #FFFFFF !important;
  border-radius: 10px !important;
  border: 2px solid #4F46E5 !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.1) !important;
}

[data-testid="stFileUploader"] [data-testid="stMarkdown"] p {
  color: #1E1B4B !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  margin: 0 !important;
}

/* Image preview styling - LARGER */
[data-testid="stImage"] {
  margin: 20px 0 !important;
  max-height: 280px !important;
  width: 100% !important;
}

[data-testid="stImage"] img {
  border-radius: 16px !important;
  border: 2px solid #4F46E5 !important;
  max-height: 280px !important;
  width: 100% !important;
  object-fit: contain !important;
  background: #FFFFFF !important;
  padding: 16px !important;
  box-shadow: 0 8px 20px rgba(79,70,229,0.15) !important;
}

/* Image filename display - GUARANTEED VISIBLE */
[data-testid="stImage"] + div {
  margin-top: 12px !important;
  font-size: 15px !important;
  color: #1E1B4B !important;
  font-weight: 700 !important;
  background: #F0F4FF !important;
  padding: 12px 16px !important;
  border-radius: 10px !important;
  border: 2px solid #4F46E5 !important;
  word-break: break-all !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.1) !important;
}

[data-testid="stImage"] + div p {
  color: #1E1B4B !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  margin: 0 !important;
}

/* Preview card */
.preview-box {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  border-radius: 16px;
  background: #FFFFFF;
  border: 2px solid #4F46E5;
  margin: 16px 0;
  flex-shrink: 0;
  box-shadow: 0 4px 12px rgba(79,70,229,0.1);
}

.preview-icon {
  width: 60px;
  height: 60px;
  border-radius: 16px;
  background: linear-gradient(135deg, #4F46E5, #7C3AED);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  color: white;
  flex-shrink: 0;
  box-shadow: 0 4px 10px rgba(79,70,229,0.3);
}

.preview-info {
  flex: 1;
}

.preview-label {
  font-size: 12px;
  font-weight: 700;
  color: #6B7280;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.preview-value {
  font-size: 18px;
  font-weight: 700;
  color: #1E1B4B;
  margin-bottom: 4px;
}

.preview-filename {
  font-size: 14px;
  font-weight: 600;
  color: #4F46E5;
  word-break: break-all;
  background: #F0F4FF;
  padding: 4px 10px;
  border-radius: 6px;
  display: inline-block;
}

/* ─── CANVAS SECTION ───────────────────────────────────────────────────── */
.canvas-wrap {
  border: 2px solid #C7D2FE;
  border-radius: 16px;
  overflow: hidden;
  background: #111827;
  margin: 12px 0 16px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  flex-shrink: 0;
}

.brush-control {
  background: #F9FAFB;
  border: 1px solid #E0E7FF;
  border-radius: 16px;
  padding: 16px;
  margin: 16px 0;
  flex-shrink: 0;
}

.brush-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 12px;
}

.brush-label span {
  color: #4F46E5;
  background: #EEF2FF;
  padding: 4px 12px;
  border-radius: 30px;
  font-size: 12px;
}

.clear-btn button {
  background: #FEF2F2 !important;
  color: #DC2626 !important;
  border: 1px solid #FECACA !important;
  border-radius: 30px !important;
  padding: 6px 16px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  height: 36px !important;
  margin-top: 0 !important;
}

.clear-btn button:hover {
  background: #FEE2E2 !important;
}

/* ─── BUTTONS ─────────────────────────────────────────────────────────── */
.stButton > button {
  width: 100% !important;
  height: 44px !important;
  border-radius: 30px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  margin: 8px 0 !important;
  border: none !important;
  transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
  color: white !important;
  box-shadow: 0 8px 20px rgba(79,70,229,0.3) !important;
}

.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 28px rgba(79,70,229,0.4) !important;
}

.stButton > button[kind="secondary"] {
  background: white !important;
  color: #374151 !important;
  border: 1.5px solid #E5E7EB !important;
}

.stButton > button[kind="secondary"]:hover {
  border-color: #4F46E5 !important;
  color: #4F46E5 !important;
  background: #F5F7FF !important;
}

/* ─── RESULTS ─────────────────────────────────────────────────────────── */
.results-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.res-empty { 
  text-align: center; 
  padding: 40px 20px;
  background: #F9FAFB;
  border-radius: 16px;
  margin: 20px 0;
}

.res-empty-d { 
  font-size: 64px; 
  font-weight: 900; 
  color: #E0E7FF; 
  line-height: 1; 
  margin-bottom: 16px;
}

.res-empty-m { 
  font-size: 13px; 
  color: #9CA3AF; 
  line-height: 1.6; 
}

.res-dw { 
  text-align: center; 
  padding: 16px 0 12px; 
  background: #F9FAFB;
  border-radius: 16px;
  margin-bottom: 16px;
}

.res-lbl { 
  font-size: 10px; 
  font-weight: 700; 
  letter-spacing: 0.12em; 
  text-transform: uppercase; 
  color: #9CA3AF; 
  margin-bottom: 4px; 
}

.res-d { 
  font-size: 80px; 
  font-weight: 900; 
  line-height: 1; 
  background: linear-gradient(135deg, #4F46E5, #7C3AED); 
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent; 
  background-clip: text; 
}

.cr { 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  margin: 12px 0 6px; 
}

.cl { 
  font-size: 13px; 
  font-weight: 600; 
  color: #374151; 
}

.cp { 
  font-size: 16px; 
  font-weight: 800; 
  color: #4F46E5; 
}

.ct { 
  height: 8px; 
  border-radius: 100px; 
  background: #EEF2FF; 
  overflow: hidden; 
  margin-bottom: 16px; 
  border: 1px solid #E0E7FF; 
}

.cf { 
  height: 100%; 
  border-radius: 100px; 
  background: linear-gradient(90deg, #4F46E5, #818CF8); 
}

.pl { 
  font-size: 10px; 
  font-weight: 700; 
  letter-spacing: 0.1em; 
  text-transform: uppercase; 
  color: #9CA3AF; 
  text-align: center; 
  margin: 16px 0 12px; 
}

.pr { 
  display: flex; 
  align-items: center; 
  gap: 10px; 
  margin-bottom: 6px; 
}

.pd { 
  font-size: 12px; 
  font-weight: 700; 
  color: #9CA3AF; 
  width: 24px; 
  text-align: right; 
  flex-shrink: 0; 
}

.pd-t { 
  color: #4F46E5 !important; 
}

.pt { 
  flex: 1; 
  height: 6px; 
  border-radius: 100px; 
  background: #F3F4F6; 
  overflow: hidden; 
}

.pft { 
  height: 100%; 
  border-radius: 100px; 
  background: linear-gradient(90deg, #4F46E5, #818CF8); 
}

.pfr { 
  height: 100%; 
  border-radius: 100px; 
  background: #E0E7FF; 
}

.pp { 
  font-size: 11px; 
  font-weight: 600; 
  color: #9CA3AF; 
  width: 40px; 
  text-align: right; 
  flex-shrink: 0; 
}

.pp-t { 
  color: #4F46E5 !important; 
}

.source-badge {
  text-align: center;
  margin-top: 16px;
  padding: 12px;
  background: #F9FAFB;
  border-radius: 30px;
  font-size: 11px;
  font-weight: 600;
  color: #6B7280;
  border: 1px solid #E0E7FF;
}

/* ─── STATS + FOOTER ───────────────────────────────────────────────────── */
.stats-strip {
  background: #fff; 
  border-top: 1px solid #E0E7FF; 
  padding: 20px 52px;
  display: grid; 
  grid-template-columns: repeat(4,1fr); 
  gap: 20px;
  box-shadow: 0 -4px 12px rgba(79,70,229,0.02);
}

.ss-l { 
  font-size: 11px; 
  font-weight: 700; 
  letter-spacing: 0.1em; 
  text-transform: uppercase; 
  color: #9CA3AF; 
  margin-bottom: 6px; 
}

.ss-v { 
  font-size: 18px; 
  font-weight: 800; 
  color: #1E1B4B; 
}

.afooter {
  background: #fff; 
  border-top: 1px solid #F0F0F0; 
  padding: 16px 52px;
  display: flex; 
  justify-content: space-between; 
  align-items: center;
}

.af-brand { 
  font-size: 14px; 
  font-weight: 700; 
  color: #374151; 
}

.af-links { 
  display: flex; 
  gap: 28px; 
  font-size: 13px; 
  color: #9CA3AF; 
}

.af-links span {
  cursor: pointer;
  transition: color 0.2s;
}

.af-links span:hover {
  color: #4F46E5;
}

.af-right { 
  font-size: 12px; 
  color: #9CA3AF; 
}

/* ─── UTILITIES ────────────────────────────────────────────────────────── */
[data-testid="stSlider"] { 
  padding: 0 !important; 
}

[data-testid="stSlider"] > div > div > div > div { 
  background: #4F46E5 !important; 
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATH = "model/mnist_cnn_v2_model.keras"

def build_model():
    inp = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu')(inp)
    x = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x); x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x); x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10,activation='softmax')(x)
    m = tf.keras.Model(inp,out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return m

@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    (xt,yt),(xv,yv) = tf.keras.datasets.mnist.load_data()
    xt = np.expand_dims(xt.astype('float32')/255,-1)
    xv = np.expand_dims(xv.astype('float32')/255,-1)
    dg = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,zoom_range=.1,width_shift_range=.1,height_shift_range=.1)
    dg.fit(xt)
    m = build_model()
    m.fit(dg.flow(xt,yt,batch_size=128),epochs=10,
          steps_per_epoch=len(xt)//128,validation_data=(xv,yv),
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(patience=2,factor=.5,min_lr=1e-6,verbose=0),
              tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,
                                               restore_best_weights=True,verbose=0),
          ],verbose=0)
    m.save(MODEL_PATH); return m


# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_upload(f):
    img = Image.open(f).convert("L")
    arr = np.array(img)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c = cv2.findNonZero(arr)
    if c is not None:
        x,y,w,h = cv2.boundingRect(c)
        arr = arr[y:y+h, x:x+w]
        p = max(w,h)//6
        arr = cv2.copyMakeBorder(arr,p,p,p,p,cv2.BORDER_CONSTANT,value=0)
    return cv2.resize(arr,(28,28)).astype('float32').reshape(1,28,28,1)/255

def preprocess_canvas(rgba: np.ndarray):
    """
    Canvas uses dark bg (~#111827 ≈ gray 24) with light strokes (~#EFEFEF ≈ gray 240).
    MNIST format = white digit (255) on black bg (0).
    The canvas is ALREADY in that format — do NOT invert.
    Bug that caused always-0: inverting made bg=231, strokes=15 → model saw noise → 0.
    Fix: threshold above bg level to isolate strokes cleanly.
    """
    gray = cv2.cvtColor(rgba.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    # bg ≈ 24, strokes ≈ 200-240 → threshold at 50 cleanly separates them
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    c = cv2.findNonZero(binary)
    if c is not None:
        x,y,w,h = cv2.boundingRect(c)
        if w > 4 and h > 4:
            binary = binary[y:y+h, x:x+w]
            p = max(w,h)//5
            binary = cv2.copyMakeBorder(binary,p,p,p,p,cv2.BORDER_CONSTANT,value=0)
    return cv2.resize(binary,(28,28)).reshape(1,28,28,1).astype('float32')/255


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED
# ═══════════════════════════════════════════════════════════════════════════
def render_nav(active):
    al = lambda p: "navlnk navlnk-act" if active==p else "navlnk"
    st.markdown(f"""
    <div class="topnav">
      <div class="nav-brand">
        <div class="nav-icon">⚡</div>
        DIGITAI
      </div>
     <div class="nav-links">
  <span class="{al('home')}">Home</span>
  <span class="{al('studio')}">Studio</span>
  <a href="https://github.com/huda-usman" target="_blank" class="navlnk" style="text-decoration: none;">GitHub</a>
</div>
    </div>""", unsafe_allow_html=True)

def result_html(probs, show_all):
    top = int(np.argmax(probs)); conf = float(probs[top]); cw = int(conf*100)
    idxs = list(range(10)) if show_all else sorted(range(10),key=lambda d:-probs[d])[:4]
    bars = "".join(f"""
    <div class="pr">
      <div class="pd {'pd-t' if d==top else ''}">{d}</div>
      <div class="pt"><div class="{'pft' if d==top else 'pfr'}" style="width:{round(probs[d]*100,1)}%"></div></div>
      <div class="pp {'pp-t' if d==top else ''}">{probs[d]:.2f}</div>
    </div>""" for d in idxs)
    return f"""
    <div class="res-dw"><div class="res-lbl">Predicted Digit</div><div class="res-d">{top}</div></div>
    <div class="cr"><span class="cl">Confidence</span><span class="cp">{conf:.1%}</span></div>
    <div class="ct"><div class="cf" style="width:{cw}%"></div></div>
    <div class="pl">Class Probabilities</div>{bars}"""


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME  (centered, no scroll, original layout)
# ═══════════════════════════════════════════════════════════════════════════
def page_home():
    render_nav("home")
    st.markdown("""
    <div class="home-page">
      <div class="horb1"></div>
      <div class="horb2"></div>

      <!-- LEFT -->
      <div class="home-left">
        <div class="hpill">⚡ &nbsp;POWERED BY DEEP LEARNING</div>
        <div class="hh1">Handwritten Digit</div>
        <div class="hh1g">Recognition</div>
        <div class="hsub">
          Upload or draw a digit and let our advanced CNN model predict it 
          instantly. Real-time classification with full confidence breakdown.
        </div>
        <div class="hcta">
          <a class="hbtn-primary" href="?nav=studio">🖼️ &nbsp;Upload Image</a>
          <a class="hbtn-secondary" href="?nav=studio">✏️ &nbsp;Draw Digit</a>
        </div>
        <div class="hstats">
          <div class="hstat-item">
            <div class="sv">99.55%</div>
            <div class="sl">Test Accuracy</div>
          </div>
          <div class="hstat-item">
            <div class="sv">&lt;50ms</div>
            <div class="sl">Latency</div>
          </div>
          <div class="hstat-item">
            <div class="sv">70k</div>
            <div class="sl">Training Samples</div>
          </div>
        </div>
      </div>

      <!-- RIGHT -->
      <div class="home-right">
        <div class="fcard">
          <div class="fcard-inner">
            <div class="ficon fi-a">🧠</div>
            <div>
              <div class="ft">Advanced CNN</div>
              <div class="fd">Dual conv blocks with dropout, trained with augmentation on MNIST.</div>
            </div>
          </div>
        </div>
        <div class="fcard">
          <div class="fcard-inner">
            <div class="ficon fi-b">⚡</div>
            <div>
              <div class="ft">Instant Results</div>
              <div class="fd">Inference in under 50ms with per-class probability scores.</div>
            </div>
          </div>
        </div>
        <div class="fcard">
          <div class="fcard-inner">
            <div class="ficon fi-c">🎯</div>
            <div>
              <div class="ft">99.55% Accuracy</div>
              <div class="fd">Best val accuracy at epoch 9 — verified on 10k test samples.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — STUDIO
# ═══════════════════════════════════════════════════════════════════════════
def page_studio(model):
    render_nav("studio")

    st.markdown("""
    <div class="studio-hdr">
      <div class="studio-title">Digit Recognition Studio</div>
      <div class="studio-sub">Upload an image or draw a digit — the CNN classifies it instantly.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="padding:18px 52px 24px;background:#EEF2FF;">', unsafe_allow_html=True)
    col_up, col_cnv, col_res = st.columns([1, 1.1, 1], gap="medium")

    # ── UPLOAD ──────────────────────────────────────────────────────────────
    with col_up:
        st.markdown("""
        <div class="panel">
          <div class="ph">
            <div class="ph-l"><div class="phi phi-up">🖼️</div>
            <span class="ph-title">Image Upload</span></div>
          </div>
          <div class="pb">""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "img", type=["png","jpg","jpeg","bmp","gif"],
            label_visibility="collapsed", key="file_up"
        )
        if uploaded:
            st.image(Image.open(uploaded).convert("RGB"), use_container_width=True)
            try:
                probs = model.predict(preprocess_upload(uploaded), verbose=0)[0]
                st.session_state.up_result  = probs
                st.session_state.active_src = "upload"
            except Exception as e:
                st.error(f"❌ {e}")
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;padding:12px 14px;
                        border-radius:10px;background:#F9FAFB;border:1px solid #E0E7FF;">
              <div style="width:36px;height:36px;border-radius:9px;background:#EEF2FF;
                          display:flex;align-items:center;justify-content:center;
                          font-size:16px;flex-shrink:0;">🖼️</div>
              <div>
                <div style="font-size:11px;font-weight:700;color:#374151;">PREVIEW</div>
                <div style="font-size:12px;color:#9CA3AF;margin-top:1px;">No image selected</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── CANVAS ──────────────────────────────────────────────────────────────
    with col_cnv:
        hdr_l, hdr_r = st.columns([3,1])
        with hdr_l:
            st.markdown("""
            <div class="ph" style="margin-bottom:0;">
              <div class="ph-l">
                <div class="phi phi-cnv">✏️</div>
                <span class="ph-title">Drawing Canvas</span>
              </div>
            </div>""", unsafe_allow_html=True)
        with hdr_r:
            if st.button("🗑️ Clear", type="secondary", key="clr"):
                st.session_state.canvas_key    += 1
                st.session_state.draw_result    = None
                st.session_state.canvas_imgdata = None
                if st.session_state.active_src == "draw":
                    st.session_state.active_src = None
                st.rerun()

        brush = st.slider("", 8, 30, 16, key="brush", label_visibility="collapsed")

        st.markdown('<div class="canvas-wrap">', unsafe_allow_html=True)
        if HAS_CANVAS:
            canvas_result = st_canvas(
                fill_color       = "rgba(0,0,0,0)",
                stroke_width     = brush,
                stroke_color     = "#EFEFEF",
                background_color = "#111827",
                height           = 280,
                drawing_mode     = "freedraw",
                key              = f"canvas_{st.session_state.canvas_key}",
                display_toolbar  = False,
                update_streamlit = True,
            )
            # Persist drawing BEFORE button triggers rerun
            if (canvas_result is not None
                    and canvas_result.image_data is not None
                    and int(canvas_result.image_data.sum()) > 300):
                st.session_state.canvas_imgdata = canvas_result.image_data.copy()
        else:
            st.warning("Run: `pip install streamlit-drawable-canvas`")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("⚡  Run Prediction", type="primary", key="run_btn", use_container_width=True):
            img_data = st.session_state.get("canvas_imgdata")
            if img_data is None or int(img_data.sum()) < 300:
                st.warning("✏️ Draw a digit first!")
            else:
                try:
                    probs = model.predict(preprocess_canvas(img_data), verbose=0)[0]
                    st.session_state.draw_result = probs
                    st.session_state.active_src  = "draw"
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    # ── RESULTS ─────────────────────────────────────────────────────────────
    with col_res:
        st.markdown("""
        <div class="panel">
          <div class="ph">
            <div class="ph-l"><div class="phi phi-res">📊</div>
            <span class="ph-title">Classification Results</span></div>
          </div>
          <div class="pb">""", unsafe_allow_html=True)

        active = st.session_state.active_src
        probs  = (st.session_state.draw_result  if active == "draw"
                  else st.session_state.up_result if active == "upload"
                  else None)

        if probs is not None:
            st.markdown(result_html(probs, st.session_state.show_all), unsafe_allow_html=True)
            if st.button("Show less ▲" if st.session_state.show_all else "View all 10 ▼", key="tog"):
                st.session_state.show_all = not st.session_state.show_all
                st.rerun()
            src = "Drawing" if active == "draw" else "Uploaded Image"
            st.markdown(f"""<div style="text-align:center;margin-top:8px;
                font-size:11px;font-weight:600;color:#9CA3AF;">Source: {src}</div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="res-empty">
              <div class="res-empty-d">—</div>
              <div class="res-empty-m">Upload an image or draw a digit,<br>
                then press <strong>Run Prediction</strong></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-strip">
      <div><div class="ss-l">Model Architecture</div><div class="ss-v">CNN-v2 Deep</div></div>
      <div><div class="ss-l">Accuracy (Test Set)</div><div class="ss-v">99.55% Top-1</div></div>
      <div><div class="ss-l">Inference Time</div><div class="ss-v">&lt;50ms</div></div>
      <div><div class="ss-l">Training Dataset</div><div class="ss-v">MNIST · 70k samples</div></div>
    </div>
    <div class="afooter">
      <div class="af-brand">⚡ DIGITAI © 2026</div>
     <div class="af-links">
  <span>Privacy</span>
  <span>Terms</span>
  <a href="https://github.com/huda-usman" target="_blank" style="color: #9CA3AF; text-decoration: none; transition: color 0.2s;">GitHub</a>
</div>
      <div class="af-right">Hand-crafted for machine learning enthusiasts.</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTER  — query param nav so HTML buttons can trigger page switch
# ═══════════════════════════════════════════════════════════════════════════
if st.query_params.get("nav") == "studio":
    st.session_state.page = "studio"
    st.query_params.clear()
    st.rerun()
if st.query_params.get("nav") == "home":
    st.session_state.page = "home"
    st.query_params.clear()
    st.rerun()

with st.spinner(""):
    model = get_model()

if st.session_state.page == "home":
    page_home()
else:
    page_studio(model)
