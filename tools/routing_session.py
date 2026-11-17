#!/usr/bin/env python3
"""routing_session.py: chat traffic for a serving TideLM engine, so its routing history (the state
file adaptation writes: router mass per layer and expert) learns decode demand, not only prefill.

    python3 tools/routing_session.py --out DIR --hours 8 [--url http://127.0.0.1:8000] [--repo .]
                                     [--concurrency 2] [--eval-only] [--dry N]

The engine learns from whatever passes through it; this only has to look like real use. Requests
are /v1/chat/completions at the reasoning efforts a client sends, in the mix below:
  agent         a coding agent's episode in --repo (read_file / list_dir / grep, read-only, confined
                to the text files git tracks or would track: ignored files stay out), up to 8 turns
  code_context  a tracked source file and a task on it (explain, review, refactor, tests)
  code_write, math, science, writing, multilingual, summarize: requests the model itself writes
                (8 at a time, from a topic seed), then answers
Before and after: a fixed held-out set (EVAL below: greedy, effort low, 400 tokens), whose miss
rates (prompt / decode, from usage.muse) are the session's measurement. Written to DIR: log.jsonl
(one line a request: category, effort, tokens, rates, tok/s; no text), eval-before.json,
eval-after.json, summary.json. Stops at --hours, or when no request finished for 20 minutes.
Nothing leaves the machine: the repo's files go to the local engine only, and the history keeps
no text.
"""
import argparse, json, os, random, re, subprocess, sys, threading, time, urllib.request

MIX = [("agent", 28), ("code_context", 18), ("code_write", 16), ("math", 10), ("science", 8), ("writing", 8),
       ("multilingual", 6), ("summarize", 6)]
EFFORTS = [("none", 12), ("low", 46), ("medium", 27), ("high", 15)]
TOPICS = {
    "code_write": ["a Python CLI tool", "a Rust data structure", "a TypeScript React component", "a single-file HTML canvas game",
                   "a CUDA kernel", "a SQL schema and queries", "a bash script", "a Go HTTP service", "a C library function",
                   "a regex-heavy text parser", "an async Python crawler", "unit tests for an existing function", "a Rust CLI with clap",
                   "a numpy numerical routine", "a small interpreter", "a Dockerfile and compose file"],
    "math": ["competition algebra", "number theory", "probability puzzles", "geometry proofs", "combinatorics", "calculus",
             "linear algebra", "word problems with units", "inequalities", "a tricky estimate (Fermi problem)"],
    "science": ["physics intuition", "chemistry", "molecular biology", "astronomy", "climate and weather", "computer architecture",
                "machine learning theory", "economics", "medicine (general)", "materials and batteries"],
    "writing": ["a professional email", "a short story", "an essay outline", "editing a paragraph for clarity", "a product description",
                "a cover letter", "a speech", "release notes", "a poem with constraints", "rewriting text for a child"],
    "multilingual": ["Japanese", "Chinese", "Korean", "Spanish", "French", "German", "Portuguese", "Vietnamese"],
    "summarize": ["a long technical document", "meeting notes", "a research abstract", "a design document"],
}
EVAL = [
    "Write a Python function that returns the longest palindromic substring of a string, with a short explanation.",
    "Explain how a B-tree differs from a binary search tree and why databases use B-trees.",
    "A train leaves at 3:15 pm at 80 km/h; another leaves the same station at 4:00 pm at 110 km/h on the same track. When does the second catch up?",
    "Write a Rust function that parses 'key=value;key2=value2' into a HashMap<String, String>, handling empty input.",
    "What causes the seasons on Earth? Answer in one paragraph.",
    "Translate into Japanese: 'The meeting has been moved to Thursday afternoon; please bring the revised budget.'",
    "Write a haiku about debugging at 3 am.",
    "In TypeScript, write a debounce function with proper generic types.",
    "Prove that the square root of 2 is irrational.",
    "Give three concrete tips for reducing memory use in a Python data pipeline that loads large CSV files.",
    "Write a bash one-liner that finds the 10 largest files under the current directory.",
    "Explain the difference between TCP and UDP to a high-school student.",
    "Write a SQL query for the top 3 customers by total order value in the last 30 days (tables: customers, orders).",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "What is the expected number of rolls of a fair die until a six appears? Show the reasoning.",
    "Write a CUDA kernel that adds two float arrays, and the host code that launches it.",
]
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file of the repository (up to 300 lines from offset).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory of the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository's files for a regular expression (50 matches at most).",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
]
TEXT_EXT = (".rs", ".py", ".cu", ".md", ".sh", ".toml", ".txt", ".json", ".html", ".example", ".c", ".h", ".cpp", ".yml", ".yaml")


def pick(pairs):
    return random.choices([a for a, _ in pairs], [w for _, w in pairs])[0]


class Session:
    def __init__(self, a):
        self.a, self.url, self.lock = a, a.url.rstrip("/") + "/v1/chat/completions", threading.Lock()
        self.pools = {c: [] for c, _ in MIX}
        self.last_done, self.n, self.toks = time.time(), 0, [0, 0]
        root = os.path.realpath(a.repo)
        # tracked files and untracked ones git does not ignore (.env profiles and the like stay out)
        out = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True,
                             check=True).stdout.split()
        out = [f for f in out if os.path.isfile(os.path.join(root, f)) and not f.startswith(("output/", "bench/results/"))]
        self.root = root
        self.files = [f for f in out if f.endswith(TEXT_EXT) and os.path.getsize(os.path.join(root, f)) < 400_000]
        self.src = [f for f in self.files if f.endswith((".rs", ".py", ".cu", ".sh")) and os.path.getsize(os.path.join(root, f)) > 2000]
        self.docs = [f for f in self.files if f.endswith(".md") and os.path.getsize(os.path.join(root, f)) > 4000]
        self.model = a.model or json.load(urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30))["data"][0]["id"]

    def chat(self, messages, effort="low", max_tokens=4096, tools=None, temperature=None):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "reasoning_effort": effort}
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"content-type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=self.a.timeout))
        return r, time.time() - t0

    def record(self, cat, effort, r, dt, turn=0):
        u, m = r.get("usage", {}), r.get("usage", {}).get("muse", {})
        miss = m.get("prune_miss_request") or {}
        row = {"t": round(time.time()), "cat": cat, "effort": effort, "turn": turn, "prompt": u.get("prompt_tokens"),
               "out": u.get("completion_tokens"), "s": round(dt, 2), "tok_s": m.get("decode_tok_s"), "finish": r["choices"][0]["finish_reason"],
               **{k: miss.get(k) for k in ("miss_rate", "prompt_miss_rate", "decode_miss_rate")}}
        with self.lock:
            self.n += 1
            self.toks[0] += u.get("prompt_tokens") or 0
            self.toks[1] += u.get("completion_tokens") or 0
            self.last_done = time.time()
            with open(os.path.join(self.a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ prompts
    def generated(self, cat):
        with self.lock:
            if self.pools[cat]:
                return self.pools[cat].pop()
        topic = random.choice(TOPICS[cat])
        ask = {"code_write": f"programming requests about {topic}",
               "math": f"math problems in {topic}, each with a single checkable answer, some hard",
               "science": f"questions about {topic}, from curious-beginner to expert",
               "writing": f"writing tasks: {topic}, each with concrete details to work with",
               "multilingual": f"requests written entirely in {topic} (coding, everyday questions, writing, explanations)",
               "summarize": f"requests to summarize or analyze {topic}, each including the full text to work on (300-800 words, written by you)"}[cat]
        msg = (f"Write 8 different, realistic {ask}, as a user would type them to an AI assistant. Vary length, tone and difficulty. "
               "Reply with a JSON array of 8 strings and nothing else.")
        r, dt = self.chat([{"role": "user", "content": msg}], effort="none", max_tokens=6000, temperature=1.0)
        self.record("gen:" + cat, "none", r, dt)
        text = r["choices"][0]["message"].get("content") or ""
        m = re.search(r"\[.*\]", text, re.S)
        try:
            items = [s for s in json.loads(m.group(0)) if isinstance(s, str) and len(s) > 15] if m else []
        except ValueError:
            items = []
        with self.lock:
            self.pools[cat] += items[1:]
        return items[0] if items else None

    def read(self, f, limit=60_000):
        return open(os.path.join(self.root, f), encoding="utf-8", errors="replace").read()[:limit]

    # ------------------------------------------------------------------ agent tools (read-only)
    def path(self, p):
        full = os.path.realpath(os.path.join(self.root, (p or ".").lstrip("/")))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError("outside the repository")
        return full

    def tool(self, name, args):
        try:
            if name == "read_file":
                rel = os.path.relpath(self.path(args["path"]), self.root)
                if rel not in self.files:
                    return "error: not a readable text file of the repository"
                lines = self.read(rel, 2_000_000).splitlines()
                off = max(0, int(args.get("offset") or 0))
                return "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines[off:off + 300], off)) or "(empty)"
            if name == "list_dir":
                d = os.path.relpath(self.path(args.get("path")), self.root)
                pre = "" if d == "." else d + "/"
                names = sorted({f[len(pre):].split("/")[0] + ("/" if "/" in f[len(pre):] else "") for f in self.files if f.startswith(pre)})
                return "\n".join(names[:200]) or "error: no such directory"
            if name == "grep":
                rx, base = re.compile(args["pattern"]), os.path.relpath(self.path(args.get("path")), self.root)
                hits = []
                for f in self.files:
                    if base != "." and not f.startswith(base):
                        continue
                    for i, l in enumerate(self.read(f, 2_000_000).splitlines()):
                        if rx.search(l):
                            hits.append(f"{f}:{i + 1}: {l[:200]}")
                            if len(hits) >= 50:
                                return "\n".join(hits)
                return "\n".join(hits) or "(no matches)"
        except Exception as e:  # noqa: BLE001 - a tool error is the model's to handle
            return f"error: {e}"
        return "error: unknown tool"

    def agent(self, effort):
        f = random.choice(self.src)
        task = random.choice([f"Explain how {f} works and how the rest of the repository uses it.",
                              f"Find a likely bug or an unhandled edge case in {f}; read what you need, then explain it and propose a patch.",
                              f"I want to add a command-line option that changes behaviour in {f}. Find where options are parsed and outline the change.",
                              f"Write unit tests for the most important function in {f}. Look at the existing tests first.",
                              f"Where is the code that {f} depends on most? Trace the call path and summarize it.",
                              "Give me an overview of this repository's layout and its main components."])
        msgs = [{"role": "system", "content": "You are a coding agent working in a local repository. Use the tools to look before you answer."},
                {"role": "user", "content": task}]
        for turn in range(8):
            r, dt = self.chat(msgs, effort=effort, max_tokens=4096, tools=TOOLS)
            self.record("agent", effort, r, dt, turn)
            m = r["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if not calls or (r.get("usage", {}).get("prompt_tokens") or 0) > 60_000:
                return
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": self.tool(c["function"]["name"], args)[:20_000]})

    def one(self):
        cat, effort = pick(MIX), pick(EFFORTS)
        if cat == "agent":
            return self.agent(effort)
        if cat == "code_context":
            f = random.choice(self.src)
            ask = random.choice(["Explain what this file does, section by section.", "Review this code: list real problems, most serious first.",
                                 "Refactor the most complex function here for readability; show the new code.",
                                 "Write tests for this file.", "What would break if the largest function here were called concurrently?"])
            prompt = f"{ask}\n\n`{f}`:\n```\n{self.read(f, 40_000)}\n```"
        elif cat == "summarize" and random.random() < 0.5 and self.docs:
            f = random.choice(self.docs)
            prompt = f"Summarize this document for a new contributor, then list its open questions.\n\n{self.read(f, 40_000)}"
        else:
            prompt = self.generated(cat)
            if not prompt:
                return
        r, dt = self.chat([{"role": "user", "content": prompt}], effort=effort, max_tokens=8192 if effort == "high" else 4096)
        self.record(cat, effort, r, dt)

    # ------------------------------------------------------------------ runs
    def evaluate(self, name):
        rows = []
        for p in EVAL:
            r, dt = self.chat([{"role": "user", "content": p}], effort="low", max_tokens=400, temperature=0)
            m = r["usage"]["muse"]
            rows.append({"decode_tok_s": m.get("decode_tok_s"),
                         **{k: (m.get("prune_miss_request") or {}).get(k) for k in ("prompt_miss_rate", "decode_miss_rate", "missed_slots", "slots")}})
        mean = lambda k: sum(x[k] or 0 for x in rows) / len(rows)
        res = {"rows": rows, **{f"mean_{k}": round(mean(k), 4) for k in ("prompt_miss_rate", "decode_miss_rate", "decode_tok_s")}}
        json.dump(res, open(os.path.join(self.a.out, name), "w"), indent=1)
        print(f"[{name}] prompt miss {res['mean_prompt_miss_rate']:.4f}, decode miss {res['mean_decode_miss_rate']:.4f}, "
              f"{res['mean_decode_tok_s']:.1f} tok/s", flush=True)
        return res

    def worker(self, deadline, budget):
        while time.time() < deadline and (budget is None or self.n < budget):
            try:
                self.one()
            except Exception as e:  # noqa: BLE001 - one failed request does not end the night
                print(f"[error] {type(e).__name__}: {e}", flush=True)
                time.sleep(10)

    def run(self):
        a = self.a
        os.makedirs(a.out, exist_ok=True)
        before = self.evaluate("eval-before.json")
        if a.eval_only:
            return
        t0, deadline = time.time(), time.time() + a.hours * 3600
        ws = [threading.Thread(target=self.worker, args=(deadline, a.dry), daemon=True) for _ in range(a.concurrency)]
        for w in ws:
            w.start()
        while any(w.is_alive() for w in ws):
            time.sleep(30)
            if time.time() - self.last_done > 1200:
                print("[watchdog] no request finished for 20 minutes: stopping", flush=True)
                break
            if int(time.time() - t0) % 600 < 30:
                print(f"[{(time.time() - t0) / 60:.0f} min] {self.n} requests, {self.toks[0]} prompt / {self.toks[1]} generated tokens", flush=True)
        after = self.evaluate("eval-after.json")
        summary = {"requests": self.n, "prompt_tokens": self.toks[0], "generated_tokens": self.toks[1], "hours": round((time.time() - t0) / 3600, 2),
                   "before": {k: v for k, v in before.items() if k != "rows"}, "after": {k: v for k, v in after.items() if k != "rows"}}
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry", type=int, help="stop after N requests (a check of the plumbing)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed or time.time_ns())
    Session(a).run()
