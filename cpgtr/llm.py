"""LLM client for CP-GTR's real (non-demo) mode.


Providers (set CPGTR_PROVIDER, default "cerebras" as of 2026-08-08):
  poe        -- POE_API_KEY.       Free tier, but shared/variable-latency queue.
  groq       -- GROQ_API_KEY.      Free tier, dedicated LPU serving, fast, but a
                                    binding 100k-tokens/day cap in practice.
  cerebras   -- CEREBRAS_API_KEY.  Dev-tier account, ~10x the free-tier rate
                                    limit; dedicated wafer-scale serving, fast.
  openrouter -- OPENROUTER_API_KEY. Aggregator; latency depends on routed backend.
All four expose an OpenAI-compatible /chat/completions endpoint, so switching
is a base_url + model name + API key change, not a rewrite.


"""
from __future__ import annotations
import os
import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "llm_calls.jsonl"


USD_PER_1K_PROMPT_TOKENS = None
USD_PER_1K_COMPLETION_TOKENS = None

PROVIDERS = {
    "poe": {
        "base_url": "https://api.poe.com/v1",
        "api_key_env": "POE_API_KEY",
        # Verified against a real call on 2026-08-05: Poe's actual bot slug
        # is "Llama-3.3-70B-T", not the descriptive "Llama-3.3-70B-Instruct"
        # name the manuscript's prose uses -- same underlying model, Poe's
        # own naming. Check https://poe.com/ if this bot is renamed/retired.
        "default_model": "Llama-3.3-70B-T",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        # gpt-oss-120b (OpenAI's open-weight 120B model)
        "default_model": "gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
}

POE_BASE_URL = PROVIDERS["poe"]["base_url"]
DEFAULT_MODEL = PROVIDERS["poe"]["default_model"]


def load_dotenv(path: Path = ENV_FILE) -> None:

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class UsageTotals:
    """Thread-safe: multiple worker threads call .add() concurrently under
    the real extraction pipeline's ThreadPoolExecutor."""
    calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = False   # False until a real $/token rate is configured
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, prompt_tokens: int, completion_tokens: int, retries: int = 0) -> None:
        with self._lock:
            self.calls += 1
            self.retries += retries
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            if USD_PER_1K_PROMPT_TOKENS is not None and USD_PER_1K_COMPLETION_TOKENS is not None:
                self.cost_known = True
                self.cost_usd += (prompt_tokens / 1000) * USD_PER_1K_PROMPT_TOKENS
                self.cost_usd += (completion_tokens / 1000) * USD_PER_1K_COMPLETION_TOKENS


# Known free-tier tokens-per-minute caps, used as RateLimiter defaults when
# make_complete() doesn't get an explicit rate_limiter. Verified 2026-08-07:
# Groq's actual error message reported "Limit 12000" TPM for
# llama-3.3-70b-versatile on the on-demand tier -- a real run at 8 concurrent
# workers blew through this almost immediately (many 429s, most 5-item
# BATCH calls also failed outright since a batch alone can approach this
# limit). Poe has no comparable published TPM cap (its bottleneck is
# per-call latency variance, not a hard token budget), so it isn't listed
# here and gets no default throttle.
DEFAULT_TPM = {"groq": 12_000}


class RateLimiter:
    """Thread-safe TOKEN-budget limiter over a rolling window (not just a
    request-count cap): decouples ThreadPoolExecutor's worker count from a
    provider's tokens-per-minute cap, which -- per real Groq errors seen in
    this repo's own runs -- is the binding constraint, not
    requests-per-minute. Callers reserve an ESTIMATED token cost at
    acquire() time (conservative: based on max_tokens, an upper bound on
    what the call could use, not the eventual actual usage) since the real
    count isn't known until the call returns and by then it's too late to
    have throttled.

    A request-count cap (`rate`) can be layered on too if a provider has a
    separate RPM limit, but token budget is the primary mechanism.
    """

    def __init__(self, tokens_per_period: int | None = None, per_seconds: float = 60.0,
                 rate: int | None = None):
        self.tokens_per_period = tokens_per_period
        self.per_seconds = per_seconds
        self.rate = rate
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._token_events: list[tuple] = []   # (timestamp, estimated_tokens)

    def acquire(self, estimated_tokens: int = 0):
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.per_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                self._token_events = [(t, n) for t, n in self._token_events if t > cutoff]
                tokens_used = sum(n for _, n in self._token_events)
                rate_ok = self.rate is None or len(self._timestamps) < self.rate
                token_ok = (self.tokens_per_period is None
                            or tokens_used + estimated_tokens <= self.tokens_per_period)
                if rate_ok and token_ok:
                    self._timestamps.append(now)
                    self._token_events.append((now, estimated_tokens))
                    return
            time.sleep(0.5)


def make_complete(provider: str | None = None, model: str | None = None,
                   temperature: float = 0.0, max_tokens: int = 1024,
                   usage: UsageTotals | None = None, log_path: Path = LOG_FILE,
                   rate_limiter: RateLimiter | None = None,
                   max_retries: int = 5, retry_backoff_s: float = 2.0):
    """Return a `complete(prompt: str) -> str` callable backed by an
    OpenAI-compatible endpoint for `provider` (env CPGTR_PROVIDER, default
    "poe"). Thread-safe: safe to call the returned `complete` concurrently
    from multiple worker threads (ThreadPoolExecutor in extract.py).

    If `rate_limiter` isn't given and `provider` has a known tokens-per-
    minute cap (DEFAULT_TPM), one is constructed automatically -- verified
    necessary: an unthrottled real run against Groq at 8 concurrent workers
    hit its 12,000 TPM cap almost immediately (many 429s, and several 5-item
    BATCH calls failed outright since one batch alone can approach the
    limit). Each call reserves an estimated token cost (prompt length / 4 as
    a rough token count, plus max_tokens as the completion budget -- a
    deliberate over-estimate, since under-reserving is what causes 429s) at
    the limiter BEFORE sending the request.

    Logs every call (prompt, response, token usage, retry count, running
    totals) to `log_path` as JSONL, appended under a lock so concurrent
    writes never interleave/corrupt a line. Retries transient errors
    (timeouts, 5xx, connection errors, 429s) up to `max_retries` times with
    exponential backoff, and reports the retry count explicitly in both the
    log record and UsageTotals -- the OpenAI SDK's own default silent
    retries are disabled (max_retries=0 on the client) so this count is
    complete, not a partial view of what actually happened.

    Raises RuntimeError with a clear message (not a cryptic SDK traceback) if
    the selected provider's API key is not set.
    """
    load_dotenv()
    provider = provider or os.environ.get("CPGTR_PROVIDER", "cerebras")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown CPGTR_PROVIDER {provider!r}; choose from {sorted(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{cfg['api_key_env']} is not set for provider {provider!r}. Copy "
            f"{REPO_ROOT / '.env.example'} to {ENV_FILE} and put your real "
            f"API key there, then re-run."
        )
    model = model or cfg["default_model"]

    if rate_limiter is None and provider in DEFAULT_TPM:
        rate_limiter = RateLimiter(tokens_per_period=DEFAULT_TPM[provider])

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"], max_retries=0)

    totals = usage if usage is not None else UsageTotals()
    LOG_DIR.mkdir(exist_ok=True)
    log_lock = threading.Lock()

    def complete(prompt: str, max_tokens: int = max_tokens) -> str:
        # `max_tokens` param shadows the outer default on purpose: callers
        # that need a bigger completion budget for one call (extract.py's
        # batch prompts, which ask for several candidates in one JSON array
        # response -- a real Groq run showed the default single-item budget
        # truncated batch responses mid-JSON) can pass it per-call without
        # needing a second client/closure. Existing callers that only ever
        # pass `prompt` are unaffected.
        if rate_limiter is not None:
            # Rough, deliberately conservative estimate (chars/4 for prompt
            # tokens is the standard ballpark for English text) -- better to
            # over-reserve and wait a bit than under-reserve and 429.
            estimated = len(prompt) // 4 + max_tokens
            rate_limiter.acquire(estimated_tokens=estimated)
        t0 = time.time()
        retries = 0
        last_err = None
        resp = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                retries += 1
                # A 429 usually tells you almost exactly how long to wait
                # (Groq's error body includes it) -- back off harder than a
                # generic transient error to avoid hammering straight back
                # into the same limit.
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                delay = retry_backoff_s * (3 ** attempt) if is_rate_limit else retry_backoff_s * (2 ** attempt)
                time.sleep(delay)
        text = resp.choices[0].message.content
        u = getattr(resp, "usage", None)
        p_tok = getattr(u, "prompt_tokens", 0) or 0
        c_tok = getattr(u, "completion_tokens", 0) or 0
        totals.add(p_tok, c_tok, retries=retries)

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": provider,
            "model": model,
            "latency_s": round(time.time() - t0, 3),
            "retries": retries,
            "prompt": prompt,
            "response": text,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
            "running_total_tokens": totals.total_tokens,
            "running_total_calls": totals.calls,
        }
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return text

    complete.usage = totals
    complete.provider = provider
    complete.model = model
    return complete


def make_poe_complete(model: str = DEFAULT_MODEL, temperature: float = 0.0,
                       usage: UsageTotals | None = None, log_path: Path = LOG_FILE):
    """Back-compat wrapper: Poe specifically, single-threaded call shape used
    by existing callers. New code should prefer make_complete()."""
    return make_complete(provider="poe", model=model, temperature=temperature,
                          usage=usage, log_path=log_path)
