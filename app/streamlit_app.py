"""
Streamlit demo: type in a review, see the predicted sentiment from both
the baseline model and the transformer model (whichever are available in
models/). Run with:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import streamlit as st

# Make src/ importable when running this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.preprocess import clean_text

st.set_page_config(page_title="Amazon Review Sentiment", page_icon="⭐")
st.title("⭐ Amazon Review Sentiment Classifier")
st.write("Type or paste a review below to see the predicted sentiment.")


@st.cache_resource
def load_baseline():
    model_path = MODELS_DIR / "baseline_model.joblib"
    vec_path = MODELS_DIR / "baseline_vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(vec_path)


@st.cache_resource
def load_transformer():
    model_dir = MODELS_DIR / "transformer_model"
    if not model_dir.exists():
        return None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


baseline_clf, vectorizer = load_baseline()
transformer_model, tokenizer = load_transformer()

if baseline_clf is None and transformer_model is None:
    st.warning(
        "No trained models found in `models/`. Run notebooks 03 and/or 04 "
        "first to train and save a model."
    )

review_text = st.text_area("Review text", height=120, placeholder="e.g. This blender is amazing, works perfectly every time!")

if st.button("Predict sentiment") and review_text.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline (TF-IDF + Logistic Regression)")
        if baseline_clf is not None:
            cleaned = clean_text(review_text)
            vec = vectorizer.transform([cleaned])
            pred = baseline_clf.predict(vec)[0]
            proba = baseline_clf.predict_proba(vec)[0]
            st.metric("Predicted sentiment", pred.upper())
            st.bar_chart(dict(zip(baseline_clf.classes_, proba)))
        else:
            st.info("Baseline model not found. Run notebook 03 first.")

    with col2:
        st.subheader("Fine-tuned DistilBERT")
        if transformer_model is not None:
            import torch
            inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = transformer_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs))
            pred_label = transformer_model.config.id2label[pred_id]
            st.metric("Predicted sentiment", pred_label.upper())
            label_probs = {transformer_model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
            st.bar_chart(label_probs)
        else:
            st.info("Transformer model not found. Run notebook 04 first.")
