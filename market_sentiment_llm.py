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
    
    text = ftfy.fix_text(text) # Fix mojibake and encoding issues
    text = unicodedata.normalize("NFKD", text) # Normalize Unicode characters (fix weird encodings)
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\S+@\S+", "", text)  # Remove emails
    text = re.sub(r"[@#]\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # Remove non-ASCII characters (emojis, symbols)
    text = re.sub(r"[‘’“”…]", "'", text) # Remove special punctuation leftover from encoding issues
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Keep only letters and spaces
    text = text.lower() # Convert to lowercase
    text = re.sub(r"\s+", " ", text).strip() # Replace spaces, newlines, tabs with a single space
    
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
    prompt = f"Classify sentiment as Positive, Neutral, or Negative:\n\n{text}\n\nOnly return sentiment."
    resp = model.generate_content(prompt)
    return resp.text.strip()

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
        sentiment = analyze_sentiment(d["text"])
        row = {
            "source": d["source"],
            "author": d["author"],
            "title": d["title"],
            "description": d["description"],
            "url": d["url"],
            "publishedAt": d["publishedAt"],
            "sentiment": sentiment
        }
        results.append(row)

        if sentiment.lower() == "negative":
            send_slack_alert(f"⚠️ Negative sentiment detected:\n{d['title'][:200]}")

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