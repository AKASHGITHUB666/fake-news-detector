import gradio as gr
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

# Prediction function
def predict_news(text):
    if not text.strip():
        return "⚠️ Please enter some news text to analyze."

    cleaned = preprocess(text)
    vect_text = vectorizer.transform([cleaned])
    prob = model.predict_proba(vect_text)[0]
    label = model.predict(vect_text)[0]
    confidence = prob[label] * 100

    result = "✅ **Real News**" if label == 1 else "❌ **Fake News**"

    # Top words
    top_words = sorted(zip(vect_text.toarray()[0], vectorizer.get_feature_names_out()), reverse=True)[:5]
    top_used = ", ".join([word for _, word in top_words if _ > 0])

    return f"""
{result}

🧠 **Confidence**: {confidence:.2f}%

🔍 **Top words considered**: {top_used if top_used else 'Not available'}
"""

# Gradio UI
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, label="📝 Paste News Text"),
    outputs=gr.Markdown(label="Prediction"),
    title="📰 Fake News Detector",
    description="Check if a news article is Real or Fake using a trained ML model.",
    article="""
💡 This tool uses a logistic regression model trained on a dataset of fake and real news headlines.

🔒 Your inputs are not stored. For educational use only.
""",
    examples=[
        ["India launches its first AI-powered train across metro cities."],
        ["Aliens have landed in New York, confirmed by multiple witnesses."],
        ["Apple releases iPhone 20 with hologram features."]
    ],
    theme="soft",
    allow_flagging="never"
)

# Launch
interface.launch(share=True)
