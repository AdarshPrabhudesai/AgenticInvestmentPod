"""
The engine behind every agent: a Gemini model + a set of tools.

The agent loop (this is what makes it "agentic"):
  1. Send Gemini its role, the task and the list of tools it may use.
  2. If Gemini asks to call tools -> we run them and send the results back.
  3. Repeat until Gemini stops asking for tools and gives its JSON decision.

Free-tier friendly:
  * Short "slow down" (429) errors -> wait a few seconds and retry that one step.
  * Model out of quota / not available -> switch to the next free model in the
    agent's fallback list and start that agent again.
"""
import concurrent.futures
import json
import re
import threading
import time

from google import genai
from google.genai import errors, types

MAX_STEPS = 8          # max research rounds per agent
MAX_SHORT_WAIT = 20    # seconds we are willing to wait before switching model instead
REQUEST_TIMEOUT = 45   # seconds: give up on a single Gemini request after this
AGENT_TIME_LIMIT = 180 # seconds: an agent stops (and reports why) after this
MAX_MODELS_TRIED = 4   # an agent tries at most this many slow-failing models
NOTIFIER = None        # optional function(message) the app sets to show live status
DEAD_MODELS = set()    # models found unusable during this analysis (shared by all agents)
QUICK_FAILS = ("daily free quota", "no free quota", "not available")


# ----------------------------------------------------------------- usage counter
# Each agent runs in its own thread, so each thread keeps its own counter.
_local = threading.local()


def start_usage():
    _local.usage = {"calls": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0,
                    "thinking_tokens": 0, "largest_input": 0, "models": {}}


def current_usage() -> dict:
    """API usage of the agent that ran last in this thread (calls and tokens)."""
    return dict(getattr(_local, "usage", None) or {})


def record_call(model, input_tokens=0, output_tokens=0, thinking_tokens=0, ok=True):
    """Add one request to the counter. Gemini reports the token counts with each reply."""
    u = getattr(_local, "usage", None)
    if u is None:
        return
    if not ok:
        u["rejected"] += 1                  # refused by Gemini (rate limit, error, timeout)
        return
    u["calls"] += 1
    u["input_tokens"] += input_tokens or 0
    u["output_tokens"] += output_tokens or 0
    u["thinking_tokens"] += thinking_tokens or 0
    u["largest_input"] = max(u["largest_input"], input_tokens or 0)
    u["models"][model] = u["models"].get(model, 0) + 1


class ModelUnavailable(Exception):
    """This model can't be used right now -> try the next model."""


class QuotaError(Exception):
    """No model in the fallback list worked."""


class AgentTimeout(Exception):
    """The agent ran out of time."""


def make_client(api_key: str) -> genai.Client:
    # The timeout stops a single slow request from hanging the whole app
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(timeout=(REQUEST_TIMEOUT + 5) * 1000))


# Each Gemini request runs in a helper thread so we can stop waiting after
# REQUEST_TIMEOUT seconds, even if the library itself keeps retrying.
_REQUESTS = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def _call_with_timeout(client, model, contents, config):
    job = _REQUESTS.submit(client.models.generate_content, model=model, contents=contents, config=config)
    try:
        return job.result(timeout=REQUEST_TIMEOUT)
    except concurrent.futures.TimeoutError:
        raise ModelUnavailable(f"no reply within {REQUEST_TIMEOUT}s") from None


def _notify(msg: str):
    if NOTIFIER:
        NOTIFIER(msg)


# ----------------------------------------------------------------- rate-limit handling
def _quota_details(err: errors.APIError):
    """Read Google's 429 error: suggested wait time and which quota was hit."""
    info = (err.details or {}).get("error", {}) if isinstance(err.details, dict) else {}
    wait, quota_ids, zero_limit = None, [], "limit: 0" in str(err)
    for d in info.get("details", []):
        if "retryDelay" in d:
            wait = float(re.sub(r"[^\d.]", "", d["retryDelay"]) or 0)
        for v in d.get("violations", []):
            quota_ids.append(v.get("quotaId", ""))
            if str(v.get("quotaValue", "")) == "0":
                zero_limit = True
    return wait, quota_ids, zero_limit


def _generate(client, model, contents, config, deadline=None):
    """One Gemini request, with short retries. Raises ModelUnavailable to trigger a model switch."""
    deadline = deadline or time.time() + AGENT_TIME_LIMIT
    for attempt in range(4):
        if time.time() > deadline:
            raise AgentTimeout()
        try:
            response = _call_with_timeout(client, model, contents, config)
            meta = getattr(response, "usage_metadata", None)
            record_call(model, getattr(meta, "prompt_token_count", 0), getattr(meta, "candidates_token_count", 0),
                        getattr(meta, "thoughts_token_count", 0))
            return response
        except ModelUnavailable:
            record_call(model, ok=False)
            raise
        except errors.APIError as e:
            record_call(model, ok=False)
            msg = str(e.message or e)
            if e.code == 429:
                wait, quota_ids, zero = _quota_details(e)
                if zero:
                    raise ModelUnavailable("no free quota on this key") from e
                if any("PerDay" in q for q in quota_ids):
                    raise ModelUnavailable("daily free quota used up") from e
                wait = wait or 5 * (attempt + 1)
                if wait > MAX_SHORT_WAIT or attempt >= 1 or time.time() + wait > deadline:   # one retry, then switch
                    raise ModelUnavailable("per-minute rate limit") from e
                _notify(f"⏳ `{model}` asked us to slow down — waiting {wait:.0f}s…")
                time.sleep(wait + 1)
            elif e.code in (500, 503, 504):                 # Google overloaded: brief backoff
                if attempt == 3:
                    raise ModelUnavailable("Google servers overloaded") from e
                time.sleep(3 * (attempt + 1))
            elif e.code == 404 or (e.code == 400 and re.search(r"not (supported|enabled|found)", msg, re.I)):
                raise ModelUnavailable(f"not available for this key/feature ({msg[:80]})") from e
            else:
                raise
        except Exception as e:                              # network timeout / dropped connection
            if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
                raise ModelUnavailable(f"no reply within {REQUEST_TIMEOUT}s") from e
            raise


# ----------------------------------------------------------------- helpers
def _run_tool(fn, args: dict, trace: list):
    """Run one tool, record it for the UI, and return a JSON-safe result."""
    start = time.time()
    try:
        result = fn(**args)
    except Exception as e:                  # let the model know the tool failed
        result = {"error": f"{type(e).__name__}: {e}"}
    result = json.loads(json.dumps(result, default=str))
    preview = json.dumps(result)
    trace.append({"tool": fn.__name__, "args": args,
                  "result_preview": preview[:300] + ("…" if len(preview) > 300 else ""),
                  "seconds": round(time.time() - start, 1)})
    return result


def _text_of(response) -> str:
    """Join the answer's text parts. Newer Gemini models also return hidden
    'thought_signature' / thinking parts; we skip those (reading response.text
    directly would print a warning about them)."""
    parts = []
    for cand in (response.candidates or [])[:1]:
        for p in (cand.content.parts if cand.content else None) or []:
            if p.text and not getattr(p, "thought", False):
                parts.append(p.text)
    return "".join(parts)


def _extract_json(text: str):
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ----------------------------------------------------------------- the agent loop
def _run_once(client, model, system, task, tools, output_format, agent_name="Agent", deadline=None):
    trace: list = []
    by_name = {t.__name__: t for t in tools}
    # Gemma models don't accept a separate system instruction -> put it in the message
    is_gemma = model.startswith("gemma")
    first_msg = f"{system}\n\n---\n\n{task}" if is_gemma else task

    def cfg(**extra):
        return types.GenerateContentConfig(system_instruction=None if is_gemma else system,
                                           temperature=0.3, **extra)

    config = cfg(tools=tools or None,
                 automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=first_msg)])]

    for step in range(1, MAX_STEPS + 1):
        _notify(f"🧠 {agent_name} is thinking (step {step}, `{model}`)…")
        response = _generate(client, model, contents, config, deadline)
        calls = response.function_calls or []
        if not calls:
            break                                            # the agent is done researching
        _notify(f"🔧 {agent_name} → " + ", ".join(f"`{c.name}`" for c in calls))
        contents.append(response.candidates[0].content)      # keep the model's turn (incl. its reasoning)
        results = [types.Part.from_function_response(
                       name=c.name,
                       response={"result": _run_tool(by_name[c.name], dict(c.args or {}), trace)}
                       if c.name in by_name else {"error": f"Unknown tool {c.name}"})
                   for c in calls]
        contents.append(types.Content(role="user", parts=results))
    else:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Stop researching now and give your final JSON decision.")]))
        response = _generate(client, model, contents, cfg(), deadline)

    decision = _extract_json(_text_of(response))
    if decision is None:
        # The model answered in prose: ask it to restate that answer as clean JSON.
        fix = _generate(client, model,
                        f"Convert this analysis into JSON with this format:\n{output_format}\n\n"
                        f"Analysis:\n{_text_of(response)}", cfg(), deadline)
        decision = _extract_json(_text_of(fix))
    if decision is None:
        raise ValueError("The model did not return a readable decision.")
    return decision, trace


def run_agent(client, models, system_prompt: str, task: str, tools: list, output_format: str,
              agent_name: str = "Agent"):
    """Run one agent, trying each model in `models` until one works.
    Returns (decision_dict, trace_list, model_used, notes)."""
    models = [models] if isinstance(models, str) else list(models)
    system = (system_prompt + "\n\nWhen you have finished researching, reply with ONLY a JSON object "
              "in exactly this format:\n" + output_format)
    notes = []
    start_usage()
    deadline = time.time() + AGENT_TIME_LIMIT
    tried = 0
    for round_ in range(2):                       # 2nd round only if every model was just rate-limited
        for model in models:
            if model in DEAD_MODELS:
                continue                          # another agent already found this one unusable
            if tried >= MAX_MODELS_TRIED or time.time() > deadline:
                break
            try:
                decision, trace = _run_once(client, model, system, task, tools, output_format,
                                            agent_name, deadline)
                return decision, trace, model, notes
            except ModelUnavailable as e:
                notes.append(f"{model}: {e}")
                if not any(q in str(e) for q in QUICK_FAILS):
                    tried += 1                        # only failures that cost time count toward the limit
                if "per-minute" not in str(e):
                    DEAD_MODELS.add(model)            # stuck, out of quota or unavailable: skip it from now on
                _notify(f"🔁 {agent_name}: `{model}` unavailable ({e}) — switching model…")
            except AgentTimeout:
                break
        time_left = deadline - time.time()
        if round_ == 1 or not notes or time_left < 75 or not all("per-minute" in n for n in notes):
            break
        _notify(f"⏳ {agent_name}: all models are rate-limited — waiting 60s, then retrying…")
        time.sleep(60)
        tried = 0
    if time.time() > deadline:
        raise AgentTimeout(f"stopped after {AGENT_TIME_LIMIT // 60} minutes without an answer "
                           f"(the AI models were slow or rate-limited). Try again in a minute, or pick a different "
                           f"model for this agent under 'Models per agent'. Tried: " + (" | ".join(notes) or model))
    raise QuotaError("No free model worked. " + (" | ".join(notes) or "All models already used up today.")
                     + " Tip: wait a minute, or pick different models under 'Models per agent'.")
