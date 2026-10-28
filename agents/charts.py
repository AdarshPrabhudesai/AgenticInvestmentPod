"""Candlestick chart for the UI (display only — no decisions here)."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick(df, ticker: str, period: str, support=None, resistance=None) -> go.Figure:
    days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}.get(period)
    v = df.tail(days) if days else df
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{ticker} daily OHLC", "Volume", "RSI (14)"))
    fig.add_trace(go.Candlestick(x=v.index, open=v["Open"], high=v["High"], low=v["Low"], close=v["Close"],
                                 name="OHLC", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v["SMA 50"], name="50-day avg", line=dict(color="#2563eb", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v["SMA 200"], name="200-day avg", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    for level, label, color in ((support, "Support (agent)", "#16a34a"), (resistance, "Resistance (agent)", "#dc2626")):
        if isinstance(level, (int, float)) and level > 0:
            fig.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label, row=1, col=1)
    colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(v["Close"], v["Open"])]
    fig.add_trace(go.Bar(x=v.index, y=v["Volume"], marker_color=colors, showlegend=False, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v["RSI"], line=dict(color="#7c3aed", width=1.5), showlegend=False, name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig
