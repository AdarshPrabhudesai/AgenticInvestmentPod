"""
Free Gemini models and which agent uses which.

Google gives every model its OWN free quota, so spreading the four agents across
different models means four times the free requests. If a model is busy or out
of quota, the agent automatically falls back to the next model in its list.

Free-tier list checked Oct 2026 — use the "Check my models" button in the app to
see exactly which ones your key can use.
"""

# id -> short description shown in the UI. Order = backup order: models that have
# answered reliably come first; newer names listed as free (but not available on
# every key) come later, so a missing or stuck one costs little time.
FREE_MODELS = {
    "gemini-flash-latest":   "Google's current best Flash (reliable)",
    "gemini-3.6-flash":      "Flash – very capable",
    "gemini-3.5-flash":      "Flash – capable",
    "gemini-3.1-flash-lite": "Flash-Lite – fast, most generous quota",
    "gemini-3.5-flash-lite": "Flash-Lite – fast, generous quota",
    "gemini-3.7-flash":      "Flash – newer, not on every key",
    "gemini-3.8-flash":      "Flash – newest, not on every key",
    "gemma-4-31b-it":        "Gemma 4 (open model) – separate free quota",
    "gemma-4-26b-a4b-it":    "Gemma 4 small (open model) – separate free quota",
}

# Default model per agent: each agent gets a different, reliable model so they
# don't share one free quota. The SEC agent reads the most text, so it gets the
# strongest model.
DEFAULT_ASSIGNMENT = {
    "SEC Filings Agent": "gemini-flash-latest",
    "Price Trend Agent": "gemini-3.6-flash",
    "News Sentiment Agent": "gemini-3.5-flash",
    "Risk Agent": "gemini-3.1-flash-lite",
}


def fallback_chain(first: str, allowed=None) -> list:
    """The agent's own model first, then the other free models as backups.
    The backup list is rotated so different agents fall back to DIFFERENT models
    (spreading the load instead of all piling onto the same one)."""
    pool = [m for m in FREE_MODELS if allowed is None or m in allowed or m == "gemini-flash-latest"]
    if first not in pool:
        return [first] + pool
    i = pool.index(first)
    return pool[i:] + pool[:i]


def list_available(client) -> list:
    """Ask Google which text-generation models this API key can use."""
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (m.name or "").replace("models/", "")
        if "generateContent" in actions and ("flash" in name or "gemma" in name or "pro" in name):
            names.append(name)
    return sorted(names)
