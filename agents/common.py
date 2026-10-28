"""
Shared pieces used by every agent.

Every agent returns an AgentReport, so the Risk Agent can read the others'
work and the UI can show all of them the same way.
"""
from dataclasses import dataclass, field

SIGNALS = ("BUY", "HOLD", "SELL")

# Every analyst agent must answer in this JSON shape (agents may add extra keys).
BASE_FORMAT = """{
  "signal": "BUY" | "HOLD" | "SELL",
  "score": <integer from -100 (strong sell) to 100 (strong buy)>,
  "confidence": <integer 0-100, how sure you are>,
  "summary": "<2-3 sentence plain-English explanation a beginner understands>",
  "reasons": ["<short reason, start with + for a positive or - for a negative>", "..."]%s
}"""


@dataclass
class AgentReport:
    name: str
    signal: str = "HOLD"
    score: int = 0
    confidence: int = 0
    summary: str = ""
    reasons: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)    # agent-specific fields from the model
    trace: list = field(default_factory=list)    # tool calls the agent made
    model: str = ""                              # Gemini model that actually answered
    notes: list = field(default_factory=list)    # models skipped (quota etc.)
    seconds: float = 0                           # how long the agent took
    usage: dict = field(default_factory=dict)    # API calls and tokens (see llm.record_call)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def for_other_agents(self) -> dict:
        """Compact version passed to the Risk Agent."""
        return {"agent": self.name, "signal": self.signal, "score": self.score,
                "confidence": self.confidence, "summary": self.summary, "reasons": self.reasons,
                **{k: v for k, v in self.extra.items() if k != "headlines"}}


def build_report(name: str, decision: dict, trace: list, model: str = "", notes=None) -> AgentReport:
    """Turn the model's JSON into an AgentReport (light clean-up only)."""
    signal = str(decision.pop("signal", decision.pop("final_signal", "HOLD"))).upper().strip()
    def as_int(v, lo, hi):
        try:
            return max(lo, min(hi, int(round(float(v)))))
        except Exception:
            return 0
    return AgentReport(
        name=name,
        signal=signal if signal in SIGNALS else "HOLD",
        score=as_int(decision.pop("score", 0), -100, 100),
        confidence=as_int(decision.pop("confidence", 0), 0, 100),
        summary=str(decision.pop("summary", "")),
        reasons=[str(r) for r in decision.pop("reasons", [])],
        extra=decision,
        trace=trace,
        model=model,
        notes=notes or [],
    )
