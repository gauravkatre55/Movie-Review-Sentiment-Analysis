import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment AI | Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Styling: cinematic background + dark glass theme
# Replace BACKGROUND_URL with your own image URL if desired.
# ---------------------------------------------------------
BACKGROUND_URL = (
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba"
    "?auto=format&fit=crop&w=2400&q=85"
)

st.markdown(
    f"""
    <style>
    /* Full-page cinematic background */
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(rgba(5, 9, 22, 0.88), rgba(8, 12, 30, 0.94)),
            url("{BACKGROUND_URL}") center center / cover fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}

    [data-testid="stMain"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }}

    /* Hero panel */
    .hero {{
        text-align: center;
        padding: 34px 22px;
        margin: 10px 0 28px 0;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 22px;
        background: linear-gradient(
            125deg,
            rgba(22, 32, 62, 0.88),
            rgba(69, 38, 105, 0.78)
        );
        box-shadow: 0 18px 55px rgba(0,0,0,0.28);
        backdrop-filter: blur(12px);
    }}

    .hero h1 {{
        color: #ffffff;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 850;
        letter-spacing: 2px;
        margin: 0 0 10px 0;
    }}

    .hero .subtitle {{
        color: #e0d7ff;
        font-size: 1.05rem;
        font-weight: 650;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }}

    .hero .description {{
        color: #d3d9ea;
        font-size: 0.96rem;
        margin: 0;
    }}

    /* Glass panels */
    .glass-card {{
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(14, 22, 43, 0.78);
        box-shadow: 0 12px 35px rgba(0,0,0,0.22);
        backdrop-filter: blur(12px);
    }}

    .result-label {{
        color: #c7d2fe;
        font-size: 0.9rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .positive {{
        color: #4ade80;
        font-size: 2rem;
        font-weight: 850;
        margin-top: 8px;
    }}

    .negative {{
        color: #fb7185;
        font-size: 2rem;
        font-weight: 850;
        margin-top: 8px;
    }}

    .muted {{
        color: #c4cbe0;
        font-size: 0.92rem;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(7, 13, 29, 0.96);
        border-right: 1px solid rgba(255,255,255,0.12);
    }}

    /* Text area */
    .stTextArea textarea {{
        background: rgba(12, 20, 40, 0.92) !important;
        color: #ffffff !important;
        border: 1px solid rgba(167,139,250,0.45) !important;
        border-radius: 13px !important;
    }}

    .stTextArea textarea:focus {{
        border: 1px solid #a78bfa !important;
        box-shadow: 0 0 0 1px #a78bfa !important;
    }}

    /* Primary button */
    .stButton > button {{
        width: 100%;
        min-height: 48px;
        color: white;
        font-weight: 750;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(90deg, #6257e8, #a855f7);
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.28);
        transition: transform 0.2s ease, filter 0.2s ease;
    }}

    .stButton > button:hover {{
        color: white;
        filter: brightness(1.12);
        transform: translateY(-2px);
        border: 0;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #aeb9d3;
        font-size: 0.85rem;
        padding: 24px 10px 8px 10px;
        margin-top: 30px;
        border-top: 1px solid rgba(255,255,255,0.14);
    }}

    /* Improve general text contrast */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{
        color: #e5eaf6;
    }}

    h2, h3 {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Model settings - MUST match the training configuration
# ---------------------------------------------------------
MAX_FEATURES = 10000
MAX_LEN = 200
MODEL_PATH = "rnn_model.keras"


@st.cache_resource
def load_resources():
    """Load the trained model and IMDB word-index mapping once."""
    trained_model = load_model(MODEL_PATH)
    imdb_word_index = imdb.get_word_index()
    return trained_model, imdb_word_index


try:
    model, word_index = load_resources()
except Exception as error:
    st.error(
        f"Could not load `{MODEL_PATH}`. Make sure the model file is "
        f"in the same folder as app.py. Details: {error}"
    )
    st.stop()


def preprocess_review(review_text):
    """
    Convert user text using the IMDB word-index convention used by
    the Keras IMDB dataset, then pad/truncate to the model's 200 steps.
    """
    tokens = review_text.lower().split()
    encoded = []

    for token in tokens:
        # Keras IMDB convention: start=1, unknown=2, offset=3
        token_id = word_index.get(token, 2) + 3

        # Match imdb.load_data(num_words=10000):
        # indices >= 10000 are treated as out-of-vocabulary.
        if token_id >= MAX_FEATURES:
            token_id = 2

        encoded.append(token_id)

    return pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre"
    )


def predict_sentiment(review_text):
    model_input = preprocess_review(review_text)
    raw_score = float(model.predict(model_input, verbose=0)[0][0])

    sentiment = "Positive" if raw_score >= 0.5 else "Negative"
    confidence = raw_score if raw_score >= 0.5 else 1.0 - raw_score

    return sentiment, raw_score, confidence


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 SENTIMENT AI")
    st.caption("MOVIE REVIEW INTELLIGENCE")
    st.divider()

    st.markdown("### About")
    st.write(
        "A movie-review classification demo powered by a "
        "Simple Recurrent Neural Network (SimpleRNN)."
    )

    st.markdown("### Model details")
    st.write("🧠 Architecture: SimpleRNN")
    st.write("📚 Dataset: IMDB movie reviews")
    st.write(f"🔤 Vocabulary limit: {MAX_FEATURES:,}")
    st.write(f"📏 Sequence length: {MAX_LEN}")
    st.write("🎯 Output: Positive / Negative")

    st.divider()
    st.caption("Developed by Gaurav Katre")

# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 SENTIMENT AI</h1>
        <div class="subtitle">INTELLIGENT MOVIE REVIEW ANALYZER</div>
        <p class="description">
            Discover the sentiment behind movie reviews using
            Deep Learning and Recurrent Neural Networks.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## ✍️ Analyze Your Movie Review")
st.write("Enter a review below. The model will classify it as positive or negative.")

review = st.text_area(
    "Movie review",
    placeholder=(
        "Example: This movie was absolutely amazing. "
        "The acting was excellent and the story was fantastic!"
    ),
    height=170,
    label_visibility="collapsed"
)

analyze = st.button("🚀  ANALYZE SENTIMENT", use_container_width=True)

if analyze:
    if not review.strip():
        st.warning("Please enter a movie review before clicking Analyze.")
    else:
        with st.spinner("Analyzing your review..."):
            sentiment, raw_score, confidence = predict_sentiment(review)

        st.markdown("## 📊 Analysis Result")

        result_col, confidence_col = st.columns(2, gap="large")

        with result_col:
            if sentiment == "Positive":
                st.markdown(
                    """
                    <div class="glass-card">
                        <div class="result-label">Predicted sentiment</div>
                        <div class="positive">😊 POSITIVE</div>
                        <p class="muted">The model classified this review as positive.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="glass-card">
                        <div class="result-label">Predicted sentiment</div>
                        <div class="negative">☹️ NEGATIVE</div>
                        <p class="muted">The model classified this review as negative.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with confidence_col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="result-label">Confidence in predicted class</div>
                    <div style="color:#c4b5fd;font-size:2rem;font-weight:850;margin-top:8px;">
                        {confidence * 100:.2f}%
                    </div>
                    <p class="muted">Raw sigmoid output: {raw_score:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("**Confidence indicator**")
        st.progress(max(0.0, min(confidence, 1.0)))

        st.caption(
            "Confidence is derived from the model's sigmoid score; "
            "it is not necessarily a calibrated probability."
        )

st.markdown(
    """
    <div class="footer">
        <strong>SENTIMENT AI</strong> &nbsp;|&nbsp; Deep Learning Project<br>
        Built with Python, TensorFlow, Keras and Streamlit<br>
        Developed by Gaurav Katre
    </div>
    """,
    unsafe_allow_html=True
)
