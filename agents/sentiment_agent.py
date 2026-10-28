"""
📰 News Sentiment Agent (Gemini)
Reads the latest headlines, judges each one in context (not word-counting),
weighs how much each story really matters, and decides BUY / HOLD / SELL.
"""
from . import tools
from .common import BASE_FORMAT, AgentReport, build_report
from .llm import run_agent

NAME = "News Sentiment Agent"

SYSTEM_PROMPT = """You are a financial news analyst. Fetch the latest news about the stock and read every
headline in context. Judge each story's sentiment for the STOCK (not just the wording: e.g.
"avoids lawsuit" is good news), and how important it is (earnings, guidance, regulation and
analyst changes matter more than generic market chatter). Ignore items that are not really about
this company. Identify the main themes and decide whether the news flow supports buying or selling."""

FORMAT = BASE_FORMAT % """,
  "overall_mood": "Positive" | "Mixed" | "Negative",
  "themes": ["<main story theme in a few words>", "..."],
  "headlines": [{"id": <id from the tool>, "sentiment": "Positive" | "Neutral" | "Negative",
                 "importance": "High" | "Medium" | "Low", "why": "<very short reason>"}, "..."]"""

TOOLS = [tools.get_news_headlines]


def run(client, models, ticker: str, company: str) -> AgentReport:
    try:
        decision, trace, used, notes = run_agent(
            client, models, SYSTEM_PROMPT,
            f"Analyse the current news about {company} (ticker {ticker}, company name '{company}') "
            f"and decide BUY, HOLD or SELL. Rate every headline you receive.",
            TOOLS, FORMAT, NAME)
        report = build_report(NAME, decision, trace, used, notes)
        # Attach URLs/sources to the model's per-headline ratings for the UI
        articles = {a["id"]: a for a in tools.news_articles(ticker, company)}
        rated = []
        for h in report.extra.get("headlines", []):
            try:
                art = articles.get(int(h.get("id")))
            except (TypeError, ValueError):
                art = None
            if art:
                rated.append({**art, **h})
        report.extra["headlines"] = rated
        return report
    except Exception as e:
        return AgentReport(name=NAME, error=f"{type(e).__name__}: {e}")
