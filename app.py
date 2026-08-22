import pickle
import time
from pathlib import Path

import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(160deg, #0f172a 0%, #1e293b 55%, #0f172a 100%);
        }
        #MainMenu, footer, header {visibility: hidden;}

        .hero {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
        }
        .hero h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero p {
            color: #94a3b8;
            font-size: 1.02rem;
            margin-top: 0;
        }

        textarea {
            background-color: #111827 !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            font-size: 1.02rem !important;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.7rem 0;
            font-weight: 700;
            font-size: 1.05rem;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(139, 92, 246, 0.35);
            color: white;
        }

        .result-card {
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            margin-top: 1.4rem;
            text-align: center;
            animation: fadeIn 0.4s ease;
            border: 1px solid;
        }
        .result-card.spam {
            background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(127,29,29,0.15));
            border-color: #ef4444;
        }
        .result-card.ham {
            background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(21,128,61,0.15));
            border-color: #22c55e;
        }
        .result-label {
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .result-label.spam { color: #f87171; }
        .result-label.ham { color: #4ade80; }
        .result-sub {
            color: #cbd5e1;
            font-size: 0.95rem;
        }

        .metric-box {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }
        .metric-box .val {
            font-size: 1.4rem;
            font-weight: 800;
            color: #e2e8f0;
        }
        .metric-box .lbl {
            color: #94a3b8;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid #1e293b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model + vectorizer (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(APP_DIR / "spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(APP_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
    load_error = None
except Exception as e:  # noqa: BLE001
    model, vectorizer = None, None
    load_error = str(e)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: text, label, confidence
if "message_text" not in st.session_state:
    st.session_state.message_text = ""

EXAMPLES = {
    "🎉 Prize scam": "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now before it expires!!!",
    "📅 Casual chat": "Hey, are we still on for lunch tomorrow at 1? Let me know!",
    "💳 Phishing": "URGENT: Your bank account has been suspended. Verify your details immediately at this link to restore access.",
    "👋 Friendly note": "Hi mom, landed safely. Will call you once I reach the hotel.",
}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Shield")
    st.caption("TF-IDF + Multinomial Naive Bayes")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["label"] == "Spam")
    ham_count = total - spam_count

    c1, c2 = st.columns(2)
    c1.metric("Spam", spam_count)
    c2.metric("Ham", ham_count)
    st.metric("Total Checked", total)

    st.markdown("---")
    st.markdown("### 🧪 Try an example")
    for label, text in EXAMPLES.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state.message_text = text

    st.markdown("---")
    if st.session_state.history and st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Spam Shield</h1>
        <p>Paste any message below and let the model tell you if it's spam or safe.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"Couldn't load model files: {load_error}")
    st.stop()

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
message = st.text_area(
    "Message",
    key="message_text",
    height=150,
    placeholder="Type or paste an SMS / email message here...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])
with col1:
    check_clicked = st.button("🔍 Check Message", use_container_width=True)
with col2:
    clear_clicked = st.button("✖️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.message_text = ""
    st.rerun()

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if check_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.3)
            X = vectorizer.transform([message])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            classes = list(model.classes_)
            spam_idx = classes.index(1) if 1 in classes else int(pred)
            spam_prob = proba[spam_idx]
            ham_prob = 1 - spam_prob

            is_spam = bool(pred == 1) if 1 in classes else bool(pred)
            confidence = spam_prob if is_spam else ham_prob

        label = "Spam" if is_spam else "Ham"
        st.session_state.history.insert(
            0, {"text": message, "label": label, "confidence": confidence}
        )

        card_class = "spam" if is_spam else "ham"
        icon = "🚨" if is_spam else "✅"
        verdict = "This looks like SPAM" if is_spam else "This looks SAFE"
        sub = "Better double-check before clicking any links." if is_spam else "No suspicious patterns detected."

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label {card_class}">{icon} {verdict}</div>
                <div class="result-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="val">{confidence*100:.1f}%</div>'
                f'<div class="lbl">Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="val">{spam_prob*100:.1f}%</div>'
                f'<div class="lbl">Spam score</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box"><div class="val">{ham_prob*100:.1f}%</div>'
                f'<div class="lbl">Ham score</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(spam_prob), text=f"Spam likelihood: {spam_prob*100:.1f}%")

# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Recent checks")
    for item in st.session_state.history[:8]:
        tag = "🚨 Spam" if item["label"] == "Spam" else "✅ Ham"
        preview = item["text"][:90] + ("..." if len(item["text"]) > 90 else "")
        st.markdown(
            f"**{tag}** · {item['confidence']*100:.1f}% confidence  \n"
            f"<span style='color:#94a3b8;'>{preview}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)