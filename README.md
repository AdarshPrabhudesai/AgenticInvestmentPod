# AIP — Agentic Investment Pod 📈

A multi-agent stock research app powered by Google Gemini.

Type a company (ticker like `MSFT` or a name like `nvidia`). Four **Gemini AI agents** research it,
each **choosing its own tools and making its own decision**. There are no hard-coded buy/sell rules.

| Agent | File | Tools it can call | What it decides |
|---|---|---|---|
| 🏛️ SEC Filings Agent | `agents/sec_agent.py` | `get_financial_statements`, `read_10k_section` (business / risk factors / MD&A), `get_recent_filings` | BUY/HOLD/SELL from the official SEC EDGAR filings |
| 📊 Price Trend Agent | `agents/price_agent.py` | `get_price_summary` (OHLC, moving averages, RSI, returns), `get_company_profile` (valuation, analyst targets) | BUY/HOLD/SELL from price action & valuation, plus trend and support/resistance levels drawn on the chart |
| 📰 News Sentiment Agent | `agents/sentiment_agent.py` | `get_news_headlines` (Yahoo Finance + Google News) | Rates every headline in context and decides BUY/HOLD/SELL |
| 🛡️ Risk Agent | `agents/risk_agent.py` | `get_risk_metrics` (volatility, drawdown, beta, VaR) + the tools above | Reads the three reports, does its own risk research, weighs/overrules the analysts and makes the **final call** |

## How an agent works (`agents/llm.py`)
1. Gemini gets a **role** (system prompt), a **task**, and a list of Python **tools**.
2. Gemini decides which tools to call; our loop runs them and sends the results back (repeat up to 8 rounds).
3. When Gemini has enough information it replies with a JSON decision (signal, score, confidence, reasons…).
4. Every tool call is recorded, and the app shows it under **"🔧 How the agent researched this"**.

`agents/tools.py` only *fetches* data (SEC EDGAR, Yahoo Finance, Google News). All judgement is Gemini's.
The SEC, Price and News agents run **in parallel** (each on its own free model); then the Risk Agent reviews them.

## Run it
```bash
cd AlphaAgent        # the project folder
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
streamlit run app.py
```
Paste your Gemini API key (free at https://aistudio.google.com/apikey) into the sidebar, or set it first with
`export GEMINI_API_KEY=...` so it's pre-filled. ### Free models — one per agent
Google gives **each model its own free quota**, so AIP spreads the agents across free models
(sidebar → **🤖 Models per agent**):

| Agent | Default model | Free backups (tried automatically, in rotated order) |
|---|---|---|
| 🏛️ SEC Filings | `gemini-3.8-flash` | 3.7-flash, 3.6-flash, 3.5-flash, flash-latest, 3.5/3.1-flash-lite, Gemma 4 |
| 📊 Price Trend | `gemini-3.6-flash` | 3.5-flash, flash-latest, flash-lites, Gemma 4, 3.8/3.7-flash |
| 📰 News Sentiment | `gemini-3.5-flash` | flash-latest, flash-lites, Gemma 4, … |
| 🛡️ Risk | `gemini-3.7-flash` | 3.6-flash, 3.5-flash, … |

- Click **🔄 Check which models my key can use** to ask Google exactly which models your key has; the lists then only use those.
- If a model is **out of quota, rate-limited or not available**, the agent switches to its next backup and tells you
  (🔁 messages, plus "Skipped …" inside each agent's research steps). A model that's out of quota for the day is skipped
  by every other agent for the rest of that analysis.
- Short "slow down" requests (≤20 s) are waited out; longer ones switch model instead. If *every* model is
  rate-limited, the agent waits 60 s once and tries again.
- Gemma 4 models are open models with a separate free quota (the system prompt is merged into the message for them).
- Free-tier list checked October 2026 — Google changes it often, so use the Check button.

## Troubleshooting
- **"No free model worked"** → every model is out of quota for today (resets midnight US Pacific) or not on your key.
  Wait, use the Check button, or enable billing in Google AI Studio for much higher limits.
- Turn off **"Run … at the same time"** to run agents one by one with live status messages.

## Files
```
app.py                  Streamlit front end
agents/llm.py           Gemini agent engine (tool loop, retries, model fallback)
agents/models.py        Free model list, per-agent defaults, fallback order
agents/tools.py         Data tools (SEC, Yahoo Finance, news)
agents/common.py        AgentReport + shared JSON format
agents/sec_agent.py     🏛️ fundamentals agent
agents/price_agent.py   📊 price/technical agent
agents/sentiment_agent.py 📰 news agent
agents/risk_agent.py    🛡️ risk + final decision agent
agents/charts.py        candlestick chart (display only)
```

⚠️ Educational project. AI can be wrong. **Not financial advice.**
