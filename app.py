app_code = '''
import requests
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import os

# ── Config ──────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "7f2b0fe6bb1f4c0681ae4481720da283")

TICKER_MAP = {
    "AAPL": "Apple",  "TSLA": "Tesla",  "GOOGL": "Google",
    "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta"
}

LABELS = ["positive", "negative", "neutral"]

# ── Load FinBERT ─────────────────────────────────────────────────────────────
print("Loading FinBERT...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
print("FinBERT ready!")

# ── Core functions ───────────────────────────────────────────────────────────
def fetch_news(ticker, num_articles=15):
    company = TICKER_MAP.get(ticker.upper(), ticker)
    url     = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params  = {
        "q":        f\'"{company}" AND (earnings OR stock OR shares OR investor OR revenue OR market)\',
        "from":     from_date,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": num_articles,
        "apiKey":   NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    articles = response.json().get("articles", [])
    return [{
        "ticker":      ticker.upper(),
        "company":     company,
        "title":       a.get("title", ""),
        "description": a.get("description", ""),
        "source":      a.get("source", {}).get("name", "Unknown"),
        "published":   a.get("publishedAt", "")[:10],
        "url":         a.get("url", "")
    } for a in articles]

def classify_headline(text):
    if not text or len(text.strip()) == 0:
        return {"label": "neutral", "confidence": 0.0, "scores": {}}
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs  = F.softmax(outputs.logits, dim=1).squeeze()
    scores = {LABELS[i]: round(probs[i].item(), 4) for i in range(3)}
    best   = max(scores, key=scores.get)
    return {"label": best, "confidence": scores[best], "scores": scores}

def analyze_all_articles(articles):
    results = []
    for article in articles:
        text   = f"{article[\'title\']}. {article.get(\'description\', \'\')}"
        result = classify_headline(text)
        results.append({**article, **{
            "sentiment":      result["label"],
            "confidence":     result["confidence"],
            "score_positive": result["scores"].get("positive", 0),
            "score_negative": result["scores"].get("negative", 0),
            "score_neutral":  result["scores"].get("neutral",  0),
        }})
    return results

def calculate_sentiment_score(df):
    scores = []
    for _, row in df.iterrows():
        if row["sentiment"] == "positive":
            scores.append(+1 * row["confidence"])
        elif row["sentiment"] == "negative":
            scores.append(-1 * row["confidence"])
        else:
            scores.append(0)
    raw     = sum(scores) / len(scores)
    final   = round(raw * 100, 1)
    verdict = ("🟢 BULLISH" if final >= 20 else
               "🔴 BEARISH" if final <= -20 else "🟡 NEUTRAL")
    strength = ("Strong" if abs(final) >= 50 else
                "Mild"   if abs(final) >= 20 else "Flat")
    return {
        "score":        final,
        "verdict":      verdict,
        "strength":     strength,
        "positive_pct": round(len(df[df.sentiment=="positive"]) / len(df) * 100, 1),
        "negative_pct": round(len(df[df.sentiment=="negative"]) / len(df) * 100, 1),
        "neutral_pct":  round(len(df[df.sentiment=="neutral"])  / len(df) * 100, 1),
        "total_articles": len(df)
    }

def run_analysis(ticker_input, num_articles):
    ticker = ticker_input.strip().upper()
    if not ticker:
        return "⚠️ Please enter a ticker.", None, None, None
    if ticker not in TICKER_MAP:
        TICKER_MAP[ticker] = ticker

    articles = fetch_news(ticker, num_articles=int(num_articles))
    if not articles:
        return f"❌ No news found for {ticker}.", None, None, None

    analyzed = analyze_all_articles(articles)
    df       = pd.DataFrame(analyzed)
    report   = calculate_sentiment_score(df)

    summary = f"""
📰  TICKER        : {ticker}
🎯  SCORE         : {report["score"]:+.1f} / 100
{report["verdict"]}  ({report["strength"]})

📈  Positive      : {report["positive_pct"]}%
📉  Negative      : {report["negative_pct"]}%
➖  Neutral       : {report["neutral_pct"]}%
📄  Articles used : {report["total_articles"]}
    """

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Positive", "Negative", "Neutral"],
        y=[report["positive_pct"], report["negative_pct"], report["neutral_pct"]],
        marker_color=["#22c55e", "#ef4444", "#f59e0b"],
        text=[f"{v}%" for v in [report["positive_pct"],
                                report["negative_pct"],
                                report["neutral_pct"]]],
        textposition="outside"
    ))
    fig_bar.update_layout(title=f"{ticker} — Sentiment Breakdown",
                          yaxis=dict(range=[0,115], gridcolor="#e5e7eb"),
                          plot_bgcolor="white", height=350)

    score = report["score"]
    gauge_color = ("#22c55e" if score >= 20 else
                   "#ef4444" if score <= -20 else "#f59e0b")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 0},
        title={"text": f"<b>{ticker} Score</b><br><span style=\'font-size:13px\'>{report[\'verdict\']}</span>"},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar":  {"color": gauge_color, "thickness": 0.3},
            "steps": [
                {"range": [-100,-20], "color": "#fee2e2"},
                {"range": [-20,  20], "color": "#fefce8"},
                {"range": [20,  100], "color": "#dcfce7"},
            ]
        }
    ))
    fig_gauge.update_layout(height=320)

    df_s       = df.sort_values("confidence", ascending=False)
    row_colors = ["#dcfce7" if s=="positive" else
                  "#fee2e2" if s=="negative" else
                  "#fefce8" for s in df_s["sentiment"]]
    fig_table = go.Figure(data=[go.Table(
        columnwidth=[380,100,100,100],
        header=dict(
            values=["<b>Headline</b>","<b>Sentiment</b>","<b>Confidence</b>","<b>Source</b>"],
            fill_color="#1e293b", font=dict(color="white", size=13),
            align="left", height=35
        ),
        cells=dict(
            values=[
                df_s["title"].str[:65].tolist(),
                df_s["sentiment"].str.upper().tolist(),
                (df_s["confidence"]*100).round(1).astype(str).add("%").tolist(),
                df_s["source"].tolist()
            ],
            fill_color=[row_colors],
            font=dict(size=12), align="left", height=30
        )
    )])
    fig_table.update_layout(
        title=f"{ticker} — Headlines",
        height=80 + len(df_s)*35,
        margin=dict(t=55,b=10,l=10,r=10)
    )

    return summary, fig_bar, fig_gauge, fig_table

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Stock News Sentiment Analyzer") as app:
    gr.Markdown("""
    # 📈 Stock News Sentiment Analyzer
    ### Powered by FinBERT · Real-time news · AI sentiment scoring
    Enter any stock ticker to get an instant AI-powered sentiment report.
    """)
    with gr.Row():
        ticker_box = gr.Textbox(label="Stock Ticker",
                                placeholder="e.g. AAPL, TSLA, MSFT",
                                scale=3)
        num_slider = gr.Slider(minimum=5, maximum=20, value=10,
                               step=5, label="Articles", scale=1)
    analyze_btn    = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
    summary_box    = gr.Textbox(label="📊 Report", lines=10, interactive=False)
    with gr.Row():
        bar_chart   = gr.Plot(label="Sentiment Breakdown")
        gauge_chart = gr.Plot(label="Sentiment Score")
    headline_table = gr.Plot(label="📰 Headlines")
    gr.Examples(
        examples=[["AAPL",10],["TSLA",10],["MSFT",10],["NVDA",15]],
        inputs=[ticker_box, num_slider],
        label="Quick examples"
    )
    gr.Markdown("---\\nBuilt with Python · FinBERT · NewsAPI · Gradio")
    analyze_btn.click(
        fn=run_analysis,
        inputs=[ticker_box, num_slider],
        outputs=[summary_box, bar_chart, gauge_chart, headline_table]
    )

if __name__ == "__main__":
    app.launch()
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created!")
