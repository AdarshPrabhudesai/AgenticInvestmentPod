"""
AIP (Agentic Investment Pod) — a multi-agent stock research assistant powered by Google Gemini.

Run it with:   streamlit run app.py

Four Gemini agents, each with its own tools and its own judgement:
  1. 🏛️ SEC Filings Agent    – reads 10-K numbers & text from SEC EDGAR
  2. 📊 Price Trend Agent    – studies OHLC price action & valuation (Yahoo Finance)
  3. 📰 News Sentiment Agent – reads and rates the latest headlines
  4. 🛡️ Risk Agent           – measures risk, weighs the other three, makes the final call

Educational project only — not financial advice.
"""
import os
import time
import queue
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
import streamlit as st

from agents import charts, llm, price_agent, risk_agent, sec_agent, sentiment_agent, tools
from agents.llm import make_client
from agents.models import DEFAULT_ASSIGNMENT, FREE_MODELS, fallback_chain, list_available

st.set_page_config(page_title="AIP · Agentic Investment Pod", page_icon="📈", layout="wide")

COLORS = {"BUY": "#16a34a", "HOLD": "#d97706", "SELL": "#dc2626"}
ICONS = {"SEC Filings Agent": "🏛️", "Price Trend Agent": "📊", "News Sentiment Agent": "📰", "Risk Agent": "🛡️"}

st.markdown("""
<style>
.verdict {border-radius:16px; padding:22px 28px; color:white; margin-bottom:8px}
.verdict h1 {color:white; margin:0; font-size:3rem; line-height:1.1}
.verdict p {margin:4px 0 0 0; font-size:1.05rem; opacity:.95}
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px; min-height:120px}
.card .title {font-size:.85rem; opacity:.7; margin-bottom:6px}
.pill {display:inline-block; color:white; font-weight:700; border-radius:999px; padding:2px 12px; font-size:.95rem}
.small {font-size:.85rem; opacity:.75}
</style>""", unsafe_allow_html=True)


def pill(signal: str) -> str:
    return f'<span class="pill" style="background:{COLORS.get(signal, "#6b7280")}">{signal}</span>'


def md(text) -> str:
    """Make AI/news text safe for Streamlit markdown. Streamlit treats text between two
    $ signs as a math formula (e.g. "price ($407) is above ($366)"), which squashes the
    words together in a math font, so every $ is escaped to show as a normal dollar sign."""
    return str(text).replace("\\$", "$").replace("$", "\\$")


def show_reasons(reasons):
    for r in reasons:
        r = r.strip()
        icon = "🟢" if r.startswith("+") else "🔴" if r.startswith("-") else "⚪"
        st.write(f"{icon} {md(r.lstrip('+- ').strip())}")


def usage_line(u: dict) -> str:
    if not u:
        return "no usage recorded"
    line = (f"{u['calls']} API calls · {u['input_tokens']:,} input + {u['output_tokens']:,} output tokens · "
            f"largest single call {u['largest_input']:,} input tokens")
    return line + (f" · {u['rejected']} rejected/retried" if u.get("rejected") else "")


def show_trace(agent):
    with st.expander(f"🔧 How the agent researched this ({len(agent.trace)} tool calls · {agent.model})"):
        st.caption(f"📈 {usage_line(getattr(agent, 'usage', {}))}")
        for n in agent.notes:
            st.caption(f"🔁 Skipped {md(n)}")
        if not agent.trace:
            st.caption("The agent answered without calling any tools.")
        for i, step in enumerate(agent.trace, 1):
            st.markdown(f"**{i}. `{step['tool']}`** `{step['args']}` · {step['seconds']}s")
            st.caption(md(step["result_preview"]))


def agent_header(agent):
    st.markdown(f"**{agent.name} decided:** {pill(agent.signal)} &nbsp; "
                f"<span class='small'>confidence {agent.confidence}%</span>", unsafe_allow_html=True)
    st.progress((agent.score + 100) / 200, text=f"Score {agent.score:+d}  (−100 strong sell ←→ +100 strong buy)")
    if agent.summary:
        st.info(md(agent.summary))


# ================================================================= sidebar
with st.sidebar:
    st.title("📈 AIP")
    st.caption("**Agentic Investment Pod** · four Gemini AI agents research a stock and reach a verdict.")
    api_key = st.text_input("🔑 Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""),
                            help="Get a free key at https://aistudio.google.com/apikey. "
                                 "It stays in this browser session only.")
    api_key = (api_key or "").strip()          # copied keys often carry a space or line break
    with st.expander("🤖 Models per agent", expanded=False):
        st.caption("Each free Gemini model has its own quota, so every agent gets a different one. "
                   "If a model is busy or out of quota, the agent automatically switches to the next one.")
        if st.button("🔄 Check which models my key can use", width="stretch"):
            if not api_key:
                st.warning("Paste your API key first.")
            else:
                try:
                    st.session_state["available"] = list_available(make_client(api_key))
                except Exception as e:
                    st.error(f"Couldn't list models: {e}")
        available = st.session_state.get("available")
        options = [m for m in FREE_MODELS if available is None or m in available or m == "gemini-flash-latest"]
        if available:
            st.success(f"Your key can use {len(options)} of the free models below.")
        options = options or list(FREE_MODELS)

        def label(m):
            return f"{m} — {FREE_MODELS[m]}" if m in FREE_MODELS else m

        assignment = {}
        for name, default in DEFAULT_ASSIGNMENT.items():
            idx = options.index(default) if default in options else 0
            assignment[name] = st.selectbox(f"{ICONS[name]} {name}", options, index=idx,
                                            format_func=label, key=f"model_{name}")
    parallel = st.toggle("Run SEC, Price & News agents at the same time", value=True,
                         help="Faster. Safe on free keys because each agent uses a different model. "
                              "Turn off to run them one by one with live status messages.")
    query = st.text_input("Company or ticker", value="AAPL", help="A ticker like MSFT or a name like 'nvidia'.")
    period = st.selectbox("Chart period", ["3mo", "6mo", "1y", "2y"], index=2)
    email = st.text_input("Your email (SEC requires one)", value="student@example.com",
                          help="SEC EDGAR asks programs to identify themselves; only sent to sec.gov.")
    run = st.button("🔍 Analyze", type="primary", width="stretch")
    st.divider()
    st.markdown("**How it works**\n\n"
                "1. 🏛️ **SEC agent** reads the annual report\n"
                "2. 📊 **Price agent** reads the chart & valuation\n"
                "3. 📰 **Sentiment agent** reads the news\n"
                "4. 🛡️ **Risk agent** checks risk, weighs all three & decides\n\n"
                "Each agent runs on its own free Gemini model, chooses its own tools and makes its own call.")
    st.caption("⚠️ Educational tool — not financial advice.")

# ================================================================= welcome screen
if not run and "result" not in st.session_state:
    st.title("Welcome to AIP, your Agentic Investment Pod 👋")
    st.write("Paste your **Gemini API key**, enter a company in the sidebar and press **Analyze**. "
             "Four AI agents will research it and explain their BUY / HOLD / SELL in plain English.")
    c = st.columns(4)
    c[0].info("🏛️ **SEC Filings**\n\nReads 10-K financials, risk factors and management's discussion.")
    c[1].info("📊 **Price Trend**\n\nStudies candlesticks, moving averages, RSI and valuation.")
    c[2].info("📰 **News Sentiment**\n\nReads every recent headline and judges what it means for the stock.")
    c[3].info("🛡️ **Risk & Verdict**\n\nMeasures risk, settles disagreements, makes the final call.")
    st.stop()

# ================================================================= run the agents
if run:
    if not api_key:
        st.error("Please paste your Gemini API key in the sidebar first.")
        st.stop()
    tools.configure(email)
    try:
        match = tools.find_company(query)
    except Exception as e:
        st.error(f"Could not reach SEC EDGAR to look up the company: {e}")
        st.stop()
    if not match:
        st.error(f"Couldn't find '{query}'. Try the ticker symbol (e.g. AAPL).")
        st.stop()
    ticker, cik, company = match
    client = make_client(api_key)

    with st.status(f"Agents are researching {company}…", expanded=True) as status:
        analysts = {}

        def report(rep, slot=None):
            analysts[rep.name] = rep
            if rep.ok:
                text = (f"✅ {ICONS[rep.name]} {rep.name} decided **{rep.signal}** after {len(rep.trace)} "
                        f"tool calls · {rep.seconds:.0f}s · `{rep.model}`")
            else:
                text = f"⚠️ {ICONS[rep.name]} {rep.name} failed after {rep.seconds:.0f}s: {rep.error[:300]}"
            (slot or st).markdown(text)

        def timed(fn, *args):
            start = time.time()
            rep = fn(*args)
            rep.seconds = time.time() - start
            rep.usage = llm.current_usage()        # calls and tokens this agent used
            return rep

        live = st.empty()      # one line that updates with what the agent is doing right now

        def show_live(msg):
            if msg.startswith(("🧠", "🔧")):
                live.caption(msg)
            else:
                st.write(msg)

        llm.DEAD_MODELS.clear()
        allowed = st.session_state.get("available")
        chains = {name: fallback_chain(m, allowed) for name, m in assignment.items()}
        if not parallel:
            # One agent at a time, with live status messages (model switches, waits)
            llm.NOTIFIER = show_live
            for mod in (sec_agent, price_agent, sentiment_agent):
                st.write(f"{ICONS[mod.NAME]} {mod.NAME} is researching with `{chains[mod.NAME][0]}`…")
                report(timed(mod.run, client, chains[mod.NAME], ticker, company))
        else:
            # Threads can't draw on the page, so they drop messages in a queue and the
            # main thread shows them every half second, with one live line per agent.
            events = queue.Queue()
            llm.NOTIFIER = events.put
            st.write("🚀 SEC, Price and News agents are researching in parallel, each on its own model…")
            mods = (sec_agent, price_agent, sentiment_agent)
            lines = {m.NAME: st.empty() for m in mods}
            last = {m.NAME: f"{ICONS[m.NAME]} {m.NAME} is starting…" for m in mods}
            started = time.time()
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = {pool.submit(timed, m.run, client, chains[m.NAME], ticker, company) for m in mods}
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    while not events.empty():
                        msg = events.get()
                        name = next((n for n in last if n in msg), None)
                        if name and msg.startswith(("🧠", "🔧")):
                            last[name] = msg
                        else:
                            st.write(msg)            # model switches and waits stay visible
                    for job in done:
                        rep = job.result()
                        report(rep, lines[rep.name])
                        last.pop(rep.name, None)
                    for name, msg in last.items():   # heartbeat: shows the agent is still working
                        lines[name].caption(f"{msg} · {time.time() - started:.0f}s")
            llm.NOTIFIER = show_live
        ordered = [analysts[n] for n in (sec_agent.NAME, price_agent.NAME, sentiment_agent.NAME)]
        st.write("🛡️ Risk agent is checking risk and weighing the three reports…")
        # Put models the analysts just used at the END of the Risk Agent's list:
        # they've spent this minute's quota, so fresh models answer faster.
        busy = {r.model for r in ordered if r.model}
        risk_chain = [m for m in chains[risk_agent.NAME] if m not in busy] + \
                     [m for m in chains[risk_agent.NAME] if m in busy]
        risk = timed(risk_agent.run, client, risk_chain, ticker, company, ordered)
        live.empty()
        if risk.ok:
            st.write(f"✅ 🛡️ Risk Agent decided **{risk.signal}** after {len(risk.trace)} tool calls · "
                     f"{risk.seconds:.0f}s · `{risk.model}`")
        status.update(label="Analysis complete ✅" if risk.ok else "Finished with problems ⚠️",
                      state="complete" if risk.ok else "error", expanded=False)

    try:
        prices = tools.get_prices(ticker, "2y")
    except Exception:
        prices = pd.DataFrame()
    st.session_state["result"] = dict(ticker=ticker, company=company, cik=cik, period=period,
                                      sec=ordered[0], price=ordered[1], news=ordered[2], risk=risk, prices=prices)

R = st.session_state["result"]
ticker, company, sec, price, news, risk, prices = (R[k] for k in ("ticker", "company", "sec", "price", "news", "risk", "prices"))

# ================================================================= header + verdict
st.markdown(f"## {company} ({ticker})")
closes = prices["Close"].dropna() if not prices.empty else pd.Series(dtype=float)
if len(closes) >= 2:
    last, prev = closes.iloc[-1], closes.iloc[-2]
    m = st.columns(3)
    m[0].metric("Last price", f"${last:,.2f}", f"{(last / prev - 1) * 100:+.2f}% today")
    m[1].metric("3-month return", f"{(last / closes.iloc[-64] - 1) * 100:+.1f}%" if len(closes) > 64 else "—")
    if risk.ok:
        m[2].metric(f"Risk level ({risk.extra.get('risk_score', '?')}/100)", risk.extra.get("risk_level", "?"))
st.caption("🤖 Models that made the decisions · " + " · ".join(
    f"{ICONS[a.name]} `{a.model}`" for a in (sec, price, news, risk) if a.ok))

if risk.ok:
    st.markdown(f"""
    <div class="verdict" style="background:{COLORS[risk.signal]}">
      <p>Final recommendation from the Risk Agent</p>
      <h1>{risk.signal}</h1>
      <p>{md(risk.summary)}</p>
      <p class="small">Confidence {risk.confidence}% · Risk {risk.extra.get('risk_level', '?')}
         · Suits: {md(risk.extra.get('suitable_for', '—'))}</p>
    </div>""", unsafe_allow_html=True)
else:
    st.error(f"The Risk Agent could not reach a final decision: {risk.error}")

cols = st.columns(4)
for col, agent in zip(cols, (sec, price, news, risk)):
    with col:
        if agent.ok:
            st.markdown(f'<div class="card"><div class="title">{ICONS[agent.name]} {agent.name}</div>'
                        f'{pill(agent.signal)}<div class="small" style="margin-top:8px">'
                        f'Score {agent.score:+d} · Confidence {agent.confidence}%<br>'
                        f'{len(agent.trace)} tool calls · <code>{agent.model}</code></div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card"><div class="title">{ICONS[agent.name]} {agent.name}</div>'
                        f'<b>Failed</b><div class="small">{md(agent.error[:140])}</div></div>', unsafe_allow_html=True)
st.write("")

# ================================================================= API usage
agents_all = (sec, price, news, risk)
with st.expander(f"📈 API usage for this analysis: {sum(getattr(a, 'usage', {}).get('calls', 0) for a in agents_all)} calls · "
                 f"{sum(getattr(a, 'usage', {}).get('input_tokens', 0) + getattr(a, 'usage', {}).get('output_tokens', 0) for a in agents_all):,} tokens"):
    rows = []
    for a in agents_all:
        u = getattr(a, 'usage', {}) or {}
        rows.append({"Agent": f"{ICONS[a.name]} {a.name}", "API calls": u.get("calls", 0),
                     "Rejected / retried": u.get("rejected", 0), "Input tokens": u.get("input_tokens", 0),
                     "Output tokens": u.get("output_tokens", 0), "Thinking tokens": u.get("thinking_tokens", 0),
                     "Largest single call (input)": u.get("largest_input", 0),
                     "Models": ", ".join(f"{m} ×{n}" for m, n in u.get("models", {}).items())})
    total = {k: sum(r[k] for r in rows) for k in ("API calls", "Rejected / retried", "Input tokens",
                                                   "Output tokens", "Thinking tokens")}
    rows.append({"Agent": "Total", **total, "Largest single call (input)": max(r["Largest single call (input)"]
                                                                              for r in rows), "Models": ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Counts come from the token numbers Gemini returns with each reply. Every research round "
               "re-sends the conversation so far, which is why input tokens add up quickly. "
               "Rejected calls (rate limits, timeouts) usually don't use quota but do cost time.")

# ================================================================= detail tabs
t_chart, t_sec, t_news, t_risk = st.tabs(["📊 Price Chart", "🏛️ SEC Filings", "📰 News Sentiment", "🛡️ Risk & Final Call"])

with t_chart:
    if not prices.empty:
        st.plotly_chart(charts.candlestick(prices, ticker, R["period"],
                                           price.extra.get("support_level"), price.extra.get("resistance_level")),
                        width="stretch")
    if price.ok:
        agent_header(price)
        c1, c2 = st.columns(2)
        c1.metric("Trend (agent's view)", price.extra.get("trend", "—"))
        c2.write(f"**Valuation:** {md(price.extra.get('valuation_view', '—'))}")
        show_reasons(price.reasons)
        show_trace(price)
    else:
        st.warning(price.error)
    with st.expander("ℹ️ How to read this chart"):
        st.write("- Each **candle** is one day: green = closed higher than it opened, red = lower; "
                 "wicks show the day's high and low.\n- **50/200-day averages** smooth out noise; price above them = uptrend.\n"
                 "- **RSI** above 70 = possibly overbought, below 30 = possibly oversold.\n"
                 "- Dashed lines are the **support/resistance** levels the agent identified.")

with t_sec:
    if sec.ok:
        agent_header(sec)
        st.subheader("What the agent found")
        show_reasons(sec.reasons)
        if sec.extra.get("key_metrics"):
            st.subheader("Key metrics the agent chose")
            st.dataframe(pd.DataFrame(list(sec.extra["key_metrics"].items()), columns=["Metric", "Value"]),
                         hide_index=True, width="stretch")
        if sec.extra.get("filing_highlights"):
            st.subheader("From the filings")
            for h in sec.extra["filing_highlights"]:
                st.write(f"📄 {md(h)}")
        show_trace(sec)
        try:
            st.subheader("Recent filings")
            st.dataframe(pd.DataFrame(tools.get_recent_filings(ticker))[["form", "filed", "description", "url"]],
                         hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("Open on SEC.gov")})
        except Exception:
            pass
    else:
        st.warning(sec.error)

with t_news:
    if news.ok:
        agent_header(news)
        c1, c2 = st.columns([1, 2])
        c1.metric("Overall mood", news.extra.get("overall_mood", "—"))
        if news.extra.get("themes"):
            c2.write("**Main themes:** " + md(" · ".join(news.extra["themes"])))
        show_reasons(news.reasons)
        heads = news.extra.get("headlines", [])
        if heads:
            counts = pd.Series([h.get("sentiment", "Neutral") for h in heads]).value_counts()
            st.bar_chart(counts.reindex(["Positive", "Neutral", "Negative"]).fillna(0), horizontal=True, color="#2563eb")
            st.subheader("Headlines, as rated by the agent")
            icon = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
            for h in heads:
                st.markdown(f"{icon.get(h.get('sentiment'), '⚪')} **[{md(h['title'])}]({h['url']})**  \n"
                            f"<span class='small'>{md(h['source'])} · {h['date']} · importance "
                            f"{h.get('importance', '—')} · {md(h.get('why', ''))}</span>", unsafe_allow_html=True)
        show_trace(news)
    else:
        st.warning(news.error)

with t_risk:
    if risk.ok:
        agent_header(risk)
        st.markdown(f"**Risk level:** {risk.extra.get('risk_level', '?')} ({risk.extra.get('risk_score', '?')}/100)")
        try:
            st.progress(int(risk.extra.get("risk_score", 0)) / 100)
        except (TypeError, ValueError):
            pass
        show_reasons(risk.reasons)
        if risk.extra.get("key_risks"):
            st.subheader("Key risks")
            for k in risk.extra["key_risks"]:
                st.write(f"⚠️ {md(k)}")
        st.subheader("How the Risk Agent weighed the analysts")
        weights = risk.extra.get("agent_weights", {})
        st.dataframe(pd.DataFrame([{"Agent": f"{ICONS[a.name]} {a.name}", "Their call": a.signal if a.ok else "failed",
                                    "Score": a.score, "Confidence": f"{a.confidence}%",
                                    "Weight given": f"{weights.get(a.name, '—')}%"} for a in (sec, price, news)]),
                     hide_index=True, width="stretch")
        st.write(f"**Disagreements:** {md(risk.extra.get('how_disagreements_were_resolved', '—'))}")
        st.write(f"**Suitable for:** {md(risk.extra.get('suitable_for', '—'))}")
        st.write(f"**What would change my mind:** {md(risk.extra.get('what_would_change_my_mind', '—'))}")
        show_trace(risk)
    else:
        st.warning(risk.error)

st.divider()
st.caption("AIP (Agentic Investment Pod) is a learning project. Data: SEC EDGAR, Yahoo Finance, Google News. Decisions by Google Gemini — "
           "AI can be wrong. Not financial advice.")
