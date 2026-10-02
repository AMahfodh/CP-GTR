"""OpenAI-compatible LLM client used by CP-GTR's real (non-demo) mode.

Role in the pipeline: Stage-1 extraction (`extract.py`), the LLM-backed parser
(`parse.py`) and the LLM-based baselines (`baselines.py`) all receive a
``complete(prompt) -> str`` callable. `make_complete()` builds that callable on
top of any OpenAI-compatible ``/chat/completions`` endpoint, adding retries,
optional token-rate limiting, exact token accounting and a JSONL call log.

Provider selection
------------------
Set the environment variable ``CPGTR_PROVIDER`` (default ``"cerebras"``) or pass
``provider=`` to `make_complete()`. Each provider reads its own API key:

  ============  ====================  =========================================
  provider      key variable          notes
  ============  ====================  =========================================
  poe           POE_API_KEY           shared queue; highly variable latency
  groq          GROQ_API_KEY          fast; free tier has a low daily token cap
  cerebras      CEREBRAS_API_KEY      fast; default provider
  openrouter    OPENROUTER_API_KEY    aggregator; latency depends on the backend
  ============  ====================  =========================================

Switching providers only changes the base URL, default model name and key.
The model actually used differs between providers (see `PROVIDERS`), so every
run should record `complete.provider` and `complete.model` alongside its
results; numbers produced with different underlying models are not directly
comparable.

Configuration
-------------
  1. Copy ``.env.example`` to ``.env`` in the repository root.
  2. Put the key for your chosen provider in ``.env``, e.g. ``CEREBRAS_API_KEY=...``.
  3. ``pip install openai``.

``.env`` is read by `load_dotenv()` (a small built-in parser, no extra
dependency); variables already exported in the shell take precedence. ``.env``
holds secrets and must never be committed; no key value appears anywhere in
the source tree.

Outputs
-------
Every call is appended to ``logs/llm_calls.jsonl`` (prompt, response, token
usage, latency, retry count, running totals). Token counts are taken from the
API's ``usage`` field. Dollar cost is computed only if
`USD_PER_1K_PROMPT_TOKENS` / `USD_PER_1K_COMPLETION_TOKENS` are set.
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

# Optional price per 1,000 tokens. Providers differ in how they bill (some use
# subscription points rather than a flat $/token rate), so no default is
# assumed. Token counts are always recorded exactly from the API's usage field;
# set these two constants to your effective rate to also get `cost_usd`.
# Left as None, cost is reported as unknown rather than guessed.
USD_PER_1K_PROMPT_TOKENS = None
USD_PER_1K_COMPLETION_TOKENS = None

PROVIDERS = {
    "poe": {
        "base_url": "https://api.poe.com/v1",
        "api_key_env": "POE_API_KEY",
        # Poe's bot slug for Llama-3.3-70B-Instruct. Check https://poe.com/
        # if the bot is renamed or retired.
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
        # gpt-oss-120b (open-weight, 120B parameters). Note that this is not a
        # Llama-3.3-70B-family model, unlike the defaults of the other
        # providers; model availability differs per account, so check
        # `client.models.list()` if a call returns 404.
        "default_model": "gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
}
# Backward-compatible aliases (Poe's values).
POE_BASE_URL = PROVIDERS["poe"]["base_url"]
DEFAULT_MODEL = PROVIDERS["poe"]["default_model"]


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Blank lines and ``#`` comments are skipped; values are only stripped of
    surrounding whitespace (no quoting rules). A variable already present in
    the environment is never overwritten, so a shell-exported key wins over
    the file. Does nothing if the file does not exist."""
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
    """Running totals of calls, retries and token usage.

    Thread-safe: worker threads call `add()` concurrently. `cost_usd` is only
    meaningful when `cost_known` is True (i.e. a per-token rate is configured).
    """
    calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = False   # False until a real $/token rate is configured
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, prompt_tokens: int, completion_tokens: int, retries: int = 0) -> None:
        """Record one completed call."""
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


# Tokens-per-minute caps used as RateLimiter defaults when make_complete() is
# not given an explicit rate_limiter. Groq's on-demand tier reports a limit of
# 12,000 TPM for llama-3.3-70b-versatile; concurrent workers exceed this almost
# immediately (HTTP 429), and a single multi-item batch prompt can approach it.
# Providers without a published cap are not listed and get no default throttle.
DEFAULT_TPM = {"groq": 12_000}


class RateLimiter:
    """Thread-safe token-budget limiter over a rolling time window.

    Decouples the number of worker threads from a provider's tokens-per-minute
    cap, which is typically the binding constraint rather than requests per
    minute. Callers reserve an estimated token cost in `acquire()` before
    sending a request (an upper bound based on ``max_tokens``, since the actual
    usage is unknown until the call returns). `acquire()` blocks until the
    reservation fits in the window.

    Args:
        tokens_per_period: token budget per window (None = unlimited).
        per_seconds: window length in seconds.
        rate: optional cap on the number of requests per window.
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
        """Block until `estimated_tokens` (and one request) fit in the window,
        then record the reservation."""
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
    """Return a ``complete(prompt, max_tokens=...) -> str`` callable.

    The callable talks to the OpenAI-compatible endpoint of `provider`
    (default: env ``CPGTR_PROVIDER``, falling back to ``"cerebras"``) and is
    safe to call concurrently from multiple worker threads. It also exposes
    ``.usage`` (a `UsageTotals`), ``.provider`` and ``.model``.

    Args:
        provider: key of `PROVIDERS`; None reads ``CPGTR_PROVIDER``.
        model: model name; None uses the provider's default.
        temperature: sampling temperature (0.0 for reproducibility).
        max_tokens: default completion budget; overridable per call.
        usage: optional shared `UsageTotals` to accumulate into.
        log_path: JSONL file every call is appended to.
        rate_limiter: optional `RateLimiter`; if omitted and the provider has
            an entry in `DEFAULT_TPM`, one is created automatically.
        max_retries: retries for transient errors (timeouts, 5xx, connection
            errors, 429s), with exponential backoff (steeper for 429s).
        retry_backoff_s: base backoff delay in seconds.

    Each call reserves an estimated token cost (prompt length // 4 plus
    ``max_tokens``, a deliberate over-estimate) at the rate limiter before the
    request is sent. The SDK's own silent retries are disabled
    (``max_retries=0`` on the client) so the retry count in the log and in
    `UsageTotals` is complete. Log lines are appended under a lock so
    concurrent writes never interleave.

    Raises:
        ValueError: unknown provider.
        RuntimeError: the provider's API key is not set.
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
        # `max_tokens` shadows the outer default on purpose: callers that need
        # a larger completion budget for one call (e.g. batch extraction
        # prompts that return several candidates as one JSON array) can pass it
        # per call without building a second client.
        if rate_limiter is not None:
            # Rough, deliberately conservative estimate (about 4 characters
            # per token): over-reserving only waits, under-reserving causes 429s.
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
                # Back off more steeply on rate-limit errors than on other
                # transient errors to avoid immediately hitting the same limit.
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
    """Backward-compatible wrapper around `make_complete()` for Poe.

    New code should call `make_complete()` directly."""
    return make_complete(provider="poe", model=model, temperature=temperature,
                          usage=usage, log_path=log_path)
