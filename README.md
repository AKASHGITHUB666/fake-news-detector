Fake News Detector 

A simple yet powerful machine learning-based app that detects whether a news article is Real or Fake using Natural Language Processing (NLP) and a Logistic Regression classifier.

Built using Python, scikit-learn, Gradio, and trained on the [Kaggle Fake News dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

✨ Features

- 🔍 Detects fake news with a clean, responsive UI
- 🧠 Trained on real-world Fake and True news samples
- ⚠️ "Flag" button to mark incorrect predictions (great for improving future models!)
- 🧪 Lightweight and fast inference using Logistic Regression
- 🖼️ Interactive frontend powered by Gradio


 `Aliens landed in New York.` | ❌ Fake |
 `Supreme Court passes new data privacy bill.` | ✅ Real |


🧠 Model Details

- Model: Logistic Regression
- Vectorizer: TF-IDF
- Accuracy: ~92% (on validation set)
- Dataset: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Training Data: `True.csv` + `Fake.csv`

How to Use

🖥️ Locally (Python 3.x required)
```bash
# 1. Clone the repo
git clone https://github.com/AKASHGITHUB666/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python fake_news_app.py
