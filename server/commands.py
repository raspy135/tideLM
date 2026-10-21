"""`@@` chat commands: control messages typed into any chat client, answered without the model.

    @@help                       the commands that work right now
    @@dream status|wake|skip|log [N]

Opt-in: nothing here runs unless DSV41_DREAM=1 (server/app.py, dream/standin.py). A message is a
command only when ALL of these hold, so a real question can never be swallowed:

  * it is the LAST message of the request and its role is "user" -- clients resend the whole
    history, so an `@@dream wake` from last night is never acted on again, and an agent loop whose
    last turn is a tool result never triggers one;
  * one of its text parts is exactly one line `@@word ...` -- `@@` directly followed by a letter,
    so a pasted unified diff (`@@ -1,3 +1,4 @@`) is not one, and nothing else in that part.
    Harnesses that wrap the user's text in extra parts (reminders, context) still match.

Unknown commands get the help text, never the model. Replies carry MARKER so the conversation log
can drop command turns from what a dream learns (server/conversation_log.py).

The registry is generic (name -> handler, help line, states it is valid in); for now it holds only
`help` and `dream`. Dream state lives in files under DREAM_DIR that dream/dream.py writes:
state.json (what a running dream is doing), dreams.jsonl (one record per night), and two control
files this module writes: wake.request (stop the running dream) and skip-YYYY-MM-DD (no dream
that night). Standard library only: the stand-in server imports it without torch.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time

ENABLED_ENV = "DSV41_DREAM"
DIR_ENV = "DSV41_DREAM_DIR"
MARKER = "⁣"  # invisible separator, prefixed to every command reply
_CMD = re.compile(r"@@([A-Za-z][\w-]*)((?:[ \t]+[^\s]+)*)[ \t]*")


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "0") == "1"


def dream_dir() -> str:
    return os.environ.get(DIR_ENV, "/models/dream")


# ------------------------------------------------------------------ parsing
def _text_parts(content):
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return []


def match(messages) -> tuple[str, list[str]] | None:
    """(command, args) if the request's last message is a user command, else None."""
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    for part in _text_parts(last.get("content")):
        m = _CMD.fullmatch(part.strip())
        if m:
            return m.group(1).lower(), m.group(2).split()
    return None


def is_command_turn(msg) -> bool:
    """A user command or a command reply, for the conversation log to drop."""
    if not isinstance(msg, dict):
        return False
    parts = _text_parts(msg.get("content"))
    if msg.get("role") == "assistant":
        return any(p.startswith(MARKER) for p in parts)
    if msg.get("role") == "user":
        return any(_CMD.fullmatch(p.strip()) for p in parts)
    return False


# ------------------------------------------------------------------ dream state files
def _read_json(name, default=None):
    try:
        with open(os.path.join(dream_dir(), name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dream_state() -> dict:
    """state.json, or idle when no dream is running (a stale file from a dead driver counts as idle)."""
    s = _read_json("state.json", {}) or {}
    if s.get("state") in ("dreaming", "booting") and time.time() - s.get("updated", 0) > 900:
        s = {**s, "state": "unknown", "note": "state.json not updated for 15 min: is dream.py running?"}
    return s or {"state": "awake"}


def dream_records(n=1) -> list[dict]:
    try:
        with open(os.path.join(dream_dir(), "dreams.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        rows = []
    return rows[-n:]


def _touch(name, text=""):
    p = os.path.join(dream_dir(), name)
    with open(p, "w") as f:
        f.write(text)
    return p


def night_of(ts: float | None = None) -> _dt.date:
    """The night a moment belongs to: noon to noon, so 23:00 and 02:00 are the same night and
    `@@dream skip` typed in the evening covers the dream after midnight."""
    return (_dt.datetime.fromtimestamp(ts if ts is not None else time.time()) - _dt.timedelta(hours=12)).date()


def skip_file(day: _dt.date | None = None) -> str:
    return f"skip-{(day or night_of()).isoformat()}"


def tonight_skipped() -> bool:
    return os.path.exists(os.path.join(dream_dir(), skip_file()))


# ------------------------------------------------------------------ formatting
def _hm(ts):
    return time.strftime("%H:%M", time.localtime(ts)) if ts else "?"


def format_state(s: dict) -> str:
    st = s.get("state", "awake")
    if st == "dreaming":
        prog = s.get("stage", "?")
        if s.get("epoch") is not None:
            prog += f", epoch {s['epoch']}/{s.get('of', '?')}"
        lines = [f"💤 TideLM is dreaming (training on recent conversations).",
                 f"Started {_hm(s.get('started'))} · stage: {prog} · back around {_hm(s.get('eta'))}"]
        if s.get("facts") is not None:
            lines.append(f"Learning {s['facts']} facts from {s.get('since', '?')} to {s.get('until', '?')}.")
        lines.append("To stop now: @@dream wake (tonight's progress is kept for tomorrow).")
        return "\n".join(lines)
    if st == "booting":
        return "☀️ Waking up: the engine is booting and will answer in a few minutes."
    if st == "unknown":
        return f"Dream state unknown: {s.get('note', '')}"
    out = ["Awake."]
    if tonight_skipped():
        out.append("No dream tonight (@@dream skip).")
    last = dream_records(1)
    if last:
        out.append("Last dream: " + summary_line(last[0]))
    return " ".join(out)


def summary_line(r: dict) -> str:
    day = r.get("date", "?")
    if r.get("outcome") == "served":
        return (f"{day}: learned {r.get('n_facts', '?')} facts, overlay {r.get('generation', '?')} serving "
                f"(drift {r.get('drift', '?')}).")
    return f"{day}: {r.get('outcome', '?')} -- {r.get('reason', '')}".rstrip(" -")


def format_record(r: dict) -> str:
    if r.get("outcome") != "served" or not r.get("facts"):
        return f"Dream {summary_line(r)}"
    lines = [f"Last dream: {r.get('date')} {_hm(r.get('started'))}–{_hm(r.get('ended'))}, "
             f"conversations {r.get('since', '?')}..{r.get('until', '?')}", "",
             f"Learned {r.get('n_facts')} facts. Top ones:"]
    for i, f in enumerate(r["facts"][:5], 1):
        lines.append(f"  {i}. [{f.get('kind')}, seen {f.get('seen', 1)}] {f.get('fact')}")
    if r.get("recall"):
        lines.append(f"Recall check  {r['recall']}")
    lines.append(f"Drift         {r.get('drift')} (gate {r.get('max_drift')}) "
                 f"{'passed' if r.get('gate_pass') else 'FAILED'}")
    lines.append(f"Overlay       {r.get('generation')} now serving ({r.get('overlay_desc', '')})")
    return "\n".join(lines)


# ------------------------------------------------------------------ commands
# where: "awake" = the engine answered it, "dreaming" = the stand-in did (dream/standin.py)
def _help(args, where):
    s = dream_state() if where == "dreaming" else {"state": "awake"}
    head = f"TideLM commands (state: {s.get('state')})"
    rows = [(f"@@{name}{(' ' + sub) if sub else ''}", text)
            for name, (fn, entries) in REGISTRY.items() for sub, text, states in entries if where in states]
    w = max(len(r[0]) for r in rows)
    return head + "\n\n" + "\n".join(f"  {c.ljust(w)}  {t}" for c, t in rows)


def _dream(args, where):
    sub = args[0].lower() if args else "status"
    if sub == "status":
        return format_state(dream_state() if where == "dreaming" else {"state": "awake"})
    if sub == "log":
        try:
            n = max(1, min(int(args[1]), 14)) if len(args) > 1 else 1
        except ValueError:
            n = 1
        rows = dream_records(n)
        if not rows:
            return "No dream recorded yet."
        return format_record(rows[-1]) if n == 1 else "\n".join(summary_line(r) for r in rows)
    if sub == "wake":
        if where != "dreaming":
            return "Not dreaming: the engine is awake."
        _touch("wake.request", str(time.time()))
        return "☀️ Waking up: training stops at the next step (progress kept), then the engine boots."
    if sub == "skip":
        _touch(skip_file(), str(time.time()))
        if where == "dreaming":
            _touch("wake.request", str(time.time()))
            return "☀️ Waking up, and no dream again tonight."
        return "OK: no dream tonight."
    return None  # unknown subcommand -> help


REGISTRY = {
    "help": (_help, [("", "this list", ("awake", "dreaming"))]),
    "dream": (_dream, [("status", "what dreaming is doing now", ("awake", "dreaming")),
                       ("log [N]", "summary of the last dream (or the last N nights)", ("awake", "dreaming")),
                       ("skip", "no dream tonight (wakes a running one)", ("awake", "dreaming")),
                       ("wake", "stop the running dream; it resumes tomorrow night", ("dreaming",))]),
}


def run(cmd: str, args: list[str], where: str) -> str:
    """The reply text for a matched command, MARKER first."""
    entry = REGISTRY.get(cmd)
    out = entry[0](args, where) if entry else None
    if out is None:
        out = f"Unknown command: @@{' '.join([cmd, *args])}\n\n" + _help([], where)
    return MARKER + out


# ------------------------------------------------------------------ OpenAI-shaped replies
def completion(text: str, model: str) -> dict:
    return {"id": "chatcmpl-cmd" + os.urandom(8).hex(), "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def stream_chunks(text: str, model: str) -> list[dict]:
    rid, created = "chatcmpl-cmd" + os.urandom(8).hex(), int(time.time())

    def ch(delta, finish=None):
        return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
    return [ch({"role": "assistant", "content": ""}), ch({"content": text}), ch({}, "stop")]
