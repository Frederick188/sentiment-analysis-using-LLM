import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
import google.generativeai as genai
from newsapi import NewsApiClient
import unicodedata
import ftfy

# Load API Keys 
load_dotenv()
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
SLACK_WEBHOOK = "YOUR_SLACK_WEBHOOK"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize News API Client
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


# Cleaning Function

def clean_text(text):
    if not text:
        return ""
    
    text = ftfy.fix_text(text)  # Fix mojibake and encoding issues
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode characters
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\S+@\S+", "", text)  # Remove emails
    text = re.sub(r"[@#]\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"[‘’“”…]", "'", text)  # Fix special punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    
    return text


# Data Fetcher
def fetch_news(query="Artificial Intelligence", limit=10):
    ai_news = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=limit
    )
    articles = ai_news.get("articles", [])
    return [
        {
            "source": a.get("source", {}).get("name", "unknown"),
            "author": a.get("author", ""),
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "publishedAt": a.get("publishedAt", ""),
            "text": clean_text((a.get("title", "") or "") + " " + (a.get("description", "") or ""))
        }
        for a in articles
    ]


# Sentiment with Gemini
def analyze_sentiment(text):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    prompt = f"""
    Analyze the sentiment of the following text and respond in strict JSON.
    Text: {text}

    Return in this format:
    {{
      "sentiment": "Positive" | "Neutral" | "Negative",
      "score": float between -1 (very negative) and +1 (very positive)
    }}
    """
    resp = model.generate_content(prompt)

    try:
        result = json.loads(resp.text.strip())
        return result.get("sentiment", "Neutral"), result.get("score", 0.0)
    except Exception:
        return "Neutral", 0.0


# Slack Alerts
def send_slack_alert(message):
    if not SLACK_WEBHOOK:
        return
    requests.post(SLACK_WEBHOOK, json={"text": message})


# Main Pipeline
def run_pipeline(query="Artificial Intelligence", limit=10):
    data = fetch_news(query, limit)
    results = []

    for d in data:
        sentiment, score = analyze_sentiment(d["text"])
        row = {
            "source": d["source"],
            "author": d["author"],
            "title": d["title"],
            "description": d["description"],
            "url": d["url"],
            "publishedAt": d["publishedAt"],
            "sentiment": sentiment,
            "score": score
        }
        results.append(row)

        if sentiment.lower() == "negative":
            send_slack_alert(f"⚠️ Negative sentiment detected (score={score}):\n{d['title'][:200]}")

    # Save to CSV with UTF-8 BOM for Excel
    df = pd.DataFrame(results)
    df.to_csv("ai_news_sentiment_with_llm.csv", index=False, encoding="utf-8-sig")

    # Show summary in terminal
    print("\n Sentiment Analysis on AI News\n")
    print(tabulate(df.head(20), headers="keys", tablefmt="fancy_grid"))

    return df


# Run
if __name__ == "__main__":
    run_pipeline("Artificial Intelligence", limit=10)
