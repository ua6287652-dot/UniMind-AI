import html
import streamlit as st

from core.config import get_settings
from core.document_loader import load_uploaded_files
from core.vector_store import add_documents, load_vector_store, get_index_stats
from core.rag_pipeline import answer_question

st.set_page_config(
    page_title="UniMind AI | University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #080d18;
        --panel: #111a2b;
        --panel-2: #162238;
        --border: #263650;
        --text: #eef4ff;
        --muted: #9aabc4;
        --accent: #62a8ff;
        --accent-2: #8b7cff;
        --good: #35d39a;
    }
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(71, 111, 201, .16), transparent 28%),
          linear-gradient(135deg, #080d18 0%, #0b1220 55%, #0d1728 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a2c 0%, #0b1322 100%);
        border-right: 1px solid #263650;
    }
    [data-testid="stSidebar"] * { color: #eaf1ff; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--text); }
    .hero {
        padding: 30px 32px;
        border: 1px solid #2b4164;
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(28,53,91,.95), rgba(26,35,66,.94) 58%, rgba(49,42,91,.88));
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #9ecaff; font-size: .78rem; letter-spacing: .16em;
        text-transform: uppercase; font-weight: 800; margin-bottom: 10px;
    }
    .hero-title { color: #fff; font-size: 2.35rem; line-height: 1.12; font-weight: 850; margin: 0; }
    .hero-sub { color: #c1d1e9; font-size: 1rem; margin-top: 10px; margin-bottom: 0; }
    .section-label {
        color: #a9c7ef; text-transform: uppercase; letter-spacing: .12em;
        font-size: .76rem; font-weight: 800; margin: 24px 0 12px;
    }
    .metric-card {
        min-height: 142px; padding: 20px 21px; border-radius: 18px;
        background: linear-gradient(145deg, #17253b, #111b2d);
        border: 1px solid #2b405f; box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }
    .metric-label { color: #a9bad3; font-size: .88rem; font-weight: 650; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 850; margin-top: 12px; line-height: 1.1; }
    .metric-note { color: #8fa5c4; font-size: .78rem; margin-top: 8px; }
    .info-panel {
        padding: 20px 22px; border-radius: 17px; border: 1px solid #263b59;
        background: rgba(17,26,43,.88); margin-top: 16px;
    }
    .step-pill {
        display: inline-block; padding: 7px 11px; border-radius: 999px;
        color: #cfe4ff; background: #1a3152; border: 1px solid #31527d;
        font-size: .78rem; margin: 4px 5px 4px 0;
    }
    div[data-testid="stMetric"] {
        background: #17253b; border: 1px solid #2b405f;
        border-radius: 16px; padding: 16px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f2f6ff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #347fe8, #7668e8);
        color: white; border: 0; border-radius: 10px; font-weight: 750;
    }
    .stButton > button {
        border-radius: 10px; border: 1px solid #385071;
    }
    [data-testid="stFileUploader"] {
