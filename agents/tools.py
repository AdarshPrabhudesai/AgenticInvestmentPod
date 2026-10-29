"""
Data tools the agents can call.

These functions ONLY fetch and organise data — they make no judgements.
Gemini reads each function's name, type hints and docstring, then decides
by itself when to call them and what to conclude from the results.
"""
import copy
import html
import re
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache, wraps
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf

SEC_EMAIL = "student@example.com"   # set from the app sidebar via configure()


def ttl_cache(seconds: int = 600):
    """Remember a function's results for `seconds` (default 10 min).
    Agents often ask for the same data (e.g. the Risk Agent re-checks prices the
    Price Agent already downloaded) - this makes repeat calls instant."""
    def deco(fn):
        store, lock = {}, threading.Lock()

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                hit = store.get(key)
                if hit and time.time() - hit[0] < seconds:
                    return copy.deepcopy(hit[1])
            value = fn(*args, **kwargs)
            with lock:
                store[key] = (time.time(), value)
            return copy.deepcopy(value)
        return wrapper
    return deco


def configure(email: str):
    global SEC_EMAIL
    SEC_EMAIL = email or SEC_EMAIL


def _sec_get(url: str):
    headers = {"User-Agent": f"AIP Agentic Investment Pod research bot {SEC_EMAIL}", "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


def _num(x, digits=2):
    """Round numbers so the LLM gets clean, short values."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), digits)


# =====================================================================
#  Company lookup (shared)
# =====================================================================
@lru_cache(maxsize=1)
def _all_companies():
    return list(_sec_get("https://www.sec.gov/files/company_tickers.json").json().values())


def find_company(query: str):
    """Return (ticker, 10-digit CIK, company name) for a ticker or company name, or None."""
    q = query.strip().upper()
    companies = _all_companies()
    for c in companies:
        if c["ticker"].upper() == q:
            return c["ticker"], str(c["cik_str"]).zfill(10), c["title"]
    for c in companies:
        if q in c["title"].upper():
            return c["ticker"], str(c["cik_str"]).zfill(10), c["title"]
    return None


def _cik(ticker: str) -> str:
    match = find_company(ticker)
    if not match:
        raise ValueError(f"{ticker} not found on SEC EDGAR")
    return match[1]


# =====================================================================
#  SEC EDGAR tools
# =====================================================================
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "diluted_eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "share_buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "shareholders_equity": ["StockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
FLOW = {"revenue", "gross_profit", "operating_income", "net_income", "diluted_eps", "operating_cash_flow",
        "capital_expenditure", "research_and_development", "share_buybacks", "dividends_paid"}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "40-F"}


@ttl_cache(3600)
def _company_facts(cik: str) -> dict:
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()


def _annual(facts: dict, item: str, years: int) -> dict:
    for tag in CONCEPTS[item]:
        for tax in ("us-gaap", "ifrs-full"):
            concept = facts.get("facts", {}).get(tax, {}).get(tag)
            if not concept:
                continue
            units = concept["units"]
            rows = units.get("USD") or units.get("USD/shares") or next(iter(units.values()))
            by_end = {}
            for r in rows:
                if r.get("form") not in ANNUAL_FORMS:
                    continue
                if item in FLOW:
                    if "start" not in r:
                        continue
                    days = (datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).days
                    if not 350 <= days <= 380:
                        continue
                if r["end"] not in by_end or r["filed"] > by_end[r["end"]]["filed"]:
                    by_end[r["end"]] = r
            if by_end:
                return {e: by_end[e]["val"] for e in sorted(by_end)[-years:]}
    return {}


def get_financial_statements(ticker: str, years: int = 4) -> dict:
    """Get a company's yearly financial statement numbers from its SEC 10-K filings (XBRL data).

    Returns income statement, cash flow and balance sheet items per fiscal year,
    in US dollars (EPS in dollars per share). Missing items are null.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        years: How many recent fiscal years to return (1-6).
    """
    facts = _company_facts(_cik(ticker))
    years = max(1, min(6, int(years)))
    out = {}
    for item in CONCEPTS:
        for end, val in _annual(facts, item, years).items():
            out.setdefault(end, {})[item] = val
    return {"company": facts.get("entityName"), "units": "USD",
            "fiscal_years": {end: out[end] for end in sorted(out)[-years:]}}


@ttl_cache(1800)
def _submissions(cik: str) -> dict:
    return _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()


def get_recent_filings(ticker: str, form_type: str = "any", limit: int = 10) -> list:
    """List the company's most recent SEC filings (10-K annual, 10-Q quarterly, 8-K material events).

    Args:
        ticker: Stock ticker symbol.
        form_type: "10-K", "10-Q", "8-K" or "any".
        limit: Maximum number of filings to return.
    """
    cik = _cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    rows = []
    for i, form in enumerate(recent["form"]):
        wanted = {"10-K", "10-Q", "8-K", "20-F", "6-K"} if form_type == "any" else {form_type}
        if form not in wanted:
            continue
        acc = recent["accessionNumber"][i].replace("-", "")
        rows.append({"form": form, "filed": recent["filingDate"][i],
                     "description": recent["primaryDocDescription"][i] or form,
                     "items": recent.get("items", [""] * len(recent["form"]))[i],
                     "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{recent['primaryDocument'][i]}"})
        if len(rows) >= limit:
            break
    return rows


SECTION_PATTERNS = {
    "business": (r"item\s*1\.?\s*business", r"item\s*1a\.?\s*risk\s*factors"),
    "risk_factors": (r"item\s*1a\.?\s*risk\s*factors", r"item\s*1b\.?|item\s*1c\.?|item\s*2\.?\s*properties"),
    "mdna": (r"item\s*7\.?\s*management", r"item\s*7a\.?|item\s*8\.?\s*financial\s*statements"),
}


@ttl_cache(3600)
def _latest_10k_text(ticker: str) -> str:
    filings = get_recent_filings(ticker, "10-K", 1) or get_recent_filings(ticker, "20-F", 1)
    if not filings:
        raise ValueError("No 10-K found")
    raw = _sec_get(filings[0]["url"]).text
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text)


def read_10k_section(ticker: str, section: str, max_chars: int = 6000) -> str:
    """Read the text of a section of the company's latest 10-K annual report.

    Args:
        ticker: Stock ticker symbol.
        section: One of "business" (what the company does), "risk_factors"
            (risks management disclose) or "mdna" (Management's Discussion and
            Analysis: management's own explanation of results and outlook).
        max_chars: Maximum characters to return (up to 12000).
    """
    if section not in SECTION_PATTERNS:
        return f"Unknown section. Use one of {list(SECTION_PATTERNS)}."
    text = _latest_10k_text(ticker)
    start_pat, end_pat = SECTION_PATTERNS[section]
    # A heading appears several times (table of contents, cross-references, the real
    # section). The real section is the longest stretch that runs to the next item
    # heading without the same heading showing up again inside it.
    best, fallback = "", ""
    for m in re.finditer(start_pat, text, re.I):
        end = re.search(end_pat, text[m.end():], re.I)
        chunk = text[m.start(): m.end() + (end.start() if end else 60000)]
        if end and not re.search(start_pat, chunk[20:], re.I):
            if len(chunk) > len(best):
                best = chunk
        elif len(chunk) > len(fallback):
            fallback = chunk
    best = best if len(best) >= 500 else fallback
    if len(best) < 500:
        return "Section not found in this filing."
    return best[: min(int(max_chars), 12000)]


# =====================================================================
#  Yahoo Finance tools
# =====================================================================
@ttl_cache(600)
def get_prices(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV + indicators (used for the chart and by the tools below)."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    # Yahoo sometimes adds an unfinished row for today (or a dividend-only row) with
    # empty prices; drop those so "last price" is always a real closing price.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["SMA 50"] = df["Close"].rolling(50).mean()
    df["SMA 200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    return df


def get_price_summary(ticker: str) -> dict:
    """Get a technical snapshot of the stock's price action from Yahoo Finance.

    Includes the latest price, 50/200-day moving averages, 14-day RSI, returns over
    several horizons, 52-week high/low, volume trend and the last 20 daily closes.

    Args:
        ticker: Stock ticker symbol.
    """
    df = get_prices(ticker, "2y")
    if df.empty:
        return {"error": "no price data"}
    c, last = df["Close"], df.iloc[-1]

    def ret(days):
        return _num((c.iloc[-1] / c.iloc[-days - 1] - 1) * 100) if len(c) > days else None

    year = df.tail(252)
    return {
        "date": str(df.index[-1].date()), "last_close": _num(last["Close"]),
        "sma_50": _num(last["SMA 50"]), "sma_200": _num(last["SMA 200"]), "rsi_14": _num(last["RSI"], 1),
        "return_pct": {"1w": ret(5), "1m": ret(21), "3m": ret(63), "6m": ret(126), "1y": ret(252)},
        "high_52w": _num(year["High"].max()), "low_52w": _num(year["Low"].min()),
        "avg_volume_50d": int(df["Volume"].tail(50).mean()), "avg_volume_5d": int(df["Volume"].tail(5).mean()),
        "last_20_closes": [_num(x) for x in c.tail(20)],
    }


@ttl_cache(1800)
def get_company_profile(ticker: str) -> dict:
    """Get the company's profile and valuation from Yahoo Finance: sector, industry,
    market cap, P/E (trailing and forward), price/sales, dividend yield, analyst
    consensus and price targets. Some fields may be null.

    Args:
        ticker: Stock ticker symbol.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        return {"error": str(e)}
    keys = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
            "priceToBook", "dividendYield", "beta", "recommendationKey", "numberOfAnalystOpinions",
            "targetMeanPrice", "targetLowPrice", "targetHighPrice", "currentPrice", "longBusinessSummary"]
    out = {k: info.get(k) for k in keys}
    if out.get("longBusinessSummary"):
        out["longBusinessSummary"] = out["longBusinessSummary"][:800]
    return out


def get_risk_metrics(ticker: str) -> dict:
    """Compute objective risk statistics from the past year of daily prices:
    annualised volatility, maximum drawdown, beta and correlation versus the
    S&P 500 (SPY), 95% one-day Value-at-Risk and the worst single day.

    Args:
        ticker: Stock ticker symbol.
    """
    df = get_prices(ticker, "2y").tail(252)
    if df.empty:
        return {"error": "no price data"}
    r = df["Close"].pct_change().dropna()
    out = {
        "annual_volatility_pct": _num(r.std() * np.sqrt(252) * 100),
        "max_drawdown_1y_pct": _num(((df["Close"] / df["Close"].cummax()) - 1).min() * 100),
        "value_at_risk_95_1day_pct": _num(np.percentile(r, 5) * 100),
        "worst_day_pct": _num(r.min() * 100), "best_day_pct": _num(r.max() * 100),
    }
    try:
        spy = get_prices("SPY", "2y")["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy.rename("m")], axis=1, join="inner").dropna()
        out["beta_vs_sp500"] = _num(both["s"].cov(both["m"]) / both["m"].var())
        out["correlation_vs_sp500"] = _num(both["s"].corr(both["m"]))
        out["sp500_volatility_pct"] = _num(both["m"].std() * np.sqrt(252) * 100)
    except Exception:
        pass
    return out


# =====================================================================
#  News tools
# =====================================================================
@ttl_cache(900)
def _collect_news(ticker: str, company_name: str) -> tuple:
    items = []
    try:
        for n in yf.Ticker(ticker).news or []:
            c = n.get("content", n)
            if not c.get("title"):
                continue
            date = c.get("pubDate") or c.get("providerPublishTime") or ""
            if isinstance(date, (int, float)):
                date = datetime.fromtimestamp(date, tz=timezone.utc).isoformat()
            items.append({"title": c["title"], "summary": (c.get("summary") or "")[:400],
                          "source": (c.get("provider") or {}).get("displayName") or c.get("publisher", "Yahoo Finance"),
                          "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link", ""), "date": date[:10]})
    except Exception:
        pass
    try:
        q = quote_plus(f'"{ticker}" {company_name} stock'.strip())
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
        for e in feed.entries[:25]:
            source = e.get("source", {}).get("title", "Google News")
            title = e.get("title", "")
            if title.endswith(" - " + source):
                title = title[: -len(source) - 3]
            date = datetime(*e.published_parsed[:6]).strftime("%Y-%m-%d") if e.get("published_parsed") else ""
            items.append({"title": title, "summary": "", "source": source, "url": e.get("link", ""), "date": date})
    except Exception:
        pass
    seen, unique = set(), []
    for a in sorted(items, key=lambda a: a["date"], reverse=True):
        key = a["title"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    for i, a in enumerate(unique):
        a["id"] = i
    return tuple(unique)


def get_news_headlines(ticker: str, company_name: str = "") -> list:
    """Get recent news headlines (newest first) about the stock from Yahoo Finance
    and Google News. Each item has an id, title, short summary, source and date.

    Args:
        ticker: Stock ticker symbol.
        company_name: Company name to improve the news search (optional).
    """
    return [{k: a[k] for k in ("id", "title", "summary", "source", "date")}
            for a in _collect_news(ticker, company_name)]


def news_articles(ticker: str, company_name: str = "") -> list:
    """Full article list incl. URLs, for the UI."""
    return list(_collect_news(ticker, company_name))
