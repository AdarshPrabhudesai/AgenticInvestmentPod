"""
📊 Price Trend Agent (Gemini)
A technical/market analyst that studies the stock's OHLC price action and
valuation from Yahoo Finance and decides BUY / HOLD / SELL on its own.
(The candlestick chart itself is drawn by charts.py for the UI.)
"""
from . import tools
from .common import BASE_FORMAT, AgentReport, build_report
from .llm import run_agent

NAME = "Price Trend Agent"

SYSTEM_PROMPT = """You are a market and technical analyst. Using Yahoo Finance data, study the stock's
trend, momentum, moving averages, RSI, volume, distance from its 52-week high/low, and its
valuation versus what the market expects (P/E, analyst targets). You decide which signals matter
and how much. Explain technical terms in simple words for a beginner, and cite the numbers."""

FORMAT = BASE_FORMAT % """,
  "trend": "Uptrend" | "Downtrend" | "Sideways",
  "support_level": <price number or null>,
  "resistance_level": <price number or null>,
  "valuation_view": "<one sentence: cheap, fair or expensive and why>\""""

TOOLS = [tools.get_price_summary, tools.get_company_profile]


def run(client, models, ticker: str, company: str) -> AgentReport:
    try:
        decision, trace, used, notes = run_agent(
            client, models, SYSTEM_PROMPT,
            f"Analyse the price action and valuation of {company} ({ticker}) and decide BUY, HOLD or SELL.",
            TOOLS, FORMAT, NAME)
        return build_report(NAME, decision, trace, used, notes)
    except Exception as e:
        return AgentReport(name=NAME, error=f"{type(e).__name__}: {e}")
