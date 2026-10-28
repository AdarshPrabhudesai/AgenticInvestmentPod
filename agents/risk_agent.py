"""
🛡️ Risk Agent / Portfolio Manager (Gemini)
Reads the three analysts' reports, does its own risk research with tools,
challenges the analysts where they disagree, and makes the FINAL call.
"""
import json

from . import tools
from .common import BASE_FORMAT, AgentReport, build_report
from .llm import run_agent

NAME = "Risk Agent"

SYSTEM_PROMPT = """You are the chief risk officer and portfolio manager. Three analyst agents (SEC
fundamentals, price/technical, news sentiment) have sent you their reports. Your job:
1. Measure the stock's risk yourself with your tools (volatility, drawdown, beta, VaR, valuation,
   balance sheet). Look things up again if an analyst's claim seems doubtful.
2. Weigh the analysts' views — decide yourself how much to trust each one for THIS stock, and
   resolve disagreements explicitly.
3. Make the FINAL decision: BUY, HOLD or SELL, adjusted for risk. Explain it so a complete
   beginner understands, including what kind of investor it suits.
Be honest about uncertainty. This is educational, not personal financial advice."""

FORMAT = BASE_FORMAT % """,
  "risk_level": "Low" | "Medium" | "High",
  "risk_score": <integer 0-100, higher = riskier>,
  "key_risks": ["<specific risk>", "..."],
  "agent_weights": {"SEC Filings Agent": <percent>, "Price Trend Agent": <percent>, "News Sentiment Agent": <percent>},
  "how_disagreements_were_resolved": "<one or two sentences>",
  "suitable_for": "<what kind of investor / time horizon this suits>",
  "what_would_change_my_mind": "<event or data that would flip the call>\""""

TOOLS = [tools.get_risk_metrics, tools.get_company_profile, tools.get_financial_statements,
         tools.get_price_summary]


def run(client, models, ticker: str, company: str, analyst_reports: list) -> AgentReport:
    reports = [r.for_other_agents() if r.ok else {"agent": r.name, "error": r.error} for r in analyst_reports]
    task = (f"Stock: {company} ({ticker}).\n\nAnalyst reports:\n{json.dumps(reports, indent=2)}\n\n"
            "Research the risk, weigh the analysts, and give the final risk-adjusted decision.")
    try:
        decision, trace, used, notes = run_agent(client, models, SYSTEM_PROMPT, task, TOOLS, FORMAT, NAME)
        return build_report(NAME, decision, trace, used, notes)
    except Exception as e:
        return AgentReport(name=NAME, error=f"{type(e).__name__}: {e}")
