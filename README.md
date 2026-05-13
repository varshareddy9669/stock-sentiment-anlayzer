readme = """# 📈 Stock News Sentiment Analyzer

An end-to-end AI application that fetches real-time financial news and
classifies market sentiment using **FinBERT** — a BERT model fine-tuned
on financial text.

## 🚀 Live Demo
👉 [Try it on HuggingFace Spaces](#) ← update this link after deploying

## 🧠 How it works
1. User enters a stock ticker (e.g. AAPL, TSLA)
2. NewsAPI fetches the latest 10–20 headlines
3. FinBERT classifies each headline as Positive / Negative / Neutral
4. A weighted sentiment score (-100 to +100) is calculated
5. Results shown as gauge chart, bar chart, and color-coded headline table

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| AI Model | FinBERT (ProsusAI/finbert) |
| News Data | NewsAPI |
| Backend | Python |
| UI | Gradio |
| Charts | Plotly |
| Deployment | HuggingFace Spaces |

## 📦 Run locally
```bash
git clone https://github.com/YOUR_USERNAME/stock-sentiment-analyzer
cd stock-sentiment-analyzer
pip install -r requirements.txt
export NEWS_API_KEY=your_key_here
python app.py
```

## 📊 Features
- Real-time news fetching for any stock ticker
- AI-powered sentiment classification with confidence scores
- Interactive gauge, bar chart, and headline table
- Color-coded results (green = bullish, red = bearish)

## 👤 Author
Built by YOUR NAME as a portfolio project.
"""

with open("README.md", "w") as f:
    f.write(readme)

print("✅ README.md created!")
