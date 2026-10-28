"""
🏛️ SEC Filings Agent (Gemini)
A fundamental analyst who researches the company's official SEC filings
(10-K numbers, risk factors, management's discussion, recent 8-Ks) and decides
BUY / HOLD / SELL on its own.
"""
from . import tools
from .common import BASE_FORMAT, AgentReport, build_report
from .llm import run_agent

NAME = "SEC Filings Agent"

SYSTEM_PROMPT = """You are a senior fundamental equity analyst. You judge a company ONLY from its
official SEC EDGAR filings. Research like a professional:
- Pull the multi-year financial statements and study the trends yourself (growth, margins,
  cash generation, balance-sheet strength, returns on capital, capital allocation).
- Read at least one narrative section of the latest 10-K (MD&A and/or risk factors) to understand
  WHY the numbers look the way they do and what management expects.
- Check recent filings (e.g. 8-Ks) for material events.
Decide yourself what matters most for THIS company and its industry. Be balanced and specific:
cite actual numbers and years. Write for a beginner investor."""

FORMAT = BASE_FORMAT % """,
  "key_metrics": {"<metric name>": "<value with units, e.g. '+6.4% revenue growth (FY2025)'>", "...": "..."},
  "filing_highlights": ["<notable point you found in the 10-K text or 8-Ks>", "..."]"""

TOOLS = [tools.get_financial_statements, tools.read_10k_section, tools.get_recent_filings]


def run(client, models, ticker: str, company: str) -> AgentReport:
    try:
        decision, trace, used, notes = run_agent(
            client, models, SYSTEM_PROMPT,
            f"Analyse {company} (ticker {ticker}) using its SEC filings and decide whether it is a BUY, HOLD or SELL.",
            TOOLS, FORMAT, NAME)
        return build_report(NAME, decision, trace, used, notes)
    except Exception as e:
        return AgentReport(name=NAME, error=f"{type(e).__name__}: {e}")
