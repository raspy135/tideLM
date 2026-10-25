"""The GLM server fixture: server/muse_app.py's real HTTP handler (make_handler with GlmTemplate
and GlmParser) over a sequence of requests, a fake engine replaying each case's canned answer
as GlmTP.stream yields one (one id at a time, through the first stop id, at most max_tokens).
Records each response (status, content type, body) and the prompt ids the engine was given;
the Rust test (src/tests.rs) replays the same sequence against its own routes and compares
after normalizing ids, timestamps and the engine's stats.

    docker run --rm --user $(id -u):$(id -g) -v /home/ryan/git/tidelm:/w:ro \
        -v /home/ryan/models:/models:ro tidelm:local \
        python3 /w/rust/crates/tidelm-server/tests/gen_glm_server_cases.py /models/GLM-5.3-EXL3-3.0bpw \
        > rust/crates/tidelm-server/tests/glm_server_cases.json
"""
import copy
import json
import queue
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from server.glm_app import GlmParser, GlmTemplate  # noqa: E402
from server.muse_app import make_handler  # noqa: E402

tpl = GlmTemplate(sys.argv[1])
jobs = queue.Queue()
answer = {"ids": []}
seen = []


def fake_engine():
    while True:
        job = jobs.get()
        seen.append((job.prompt, job.params["max_tokens"]))
        n = 0
        for t in answer["ids"]:
            if n >= job.params["max_tokens"]:
                break
            job.out.put(t)
            n += 1
            if t in tpl.stop:
                break
        job.stats = {"reused_prefix": 7}
        job.out.put(None)


defaults = {"max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "reasoning_effort": "low", "max_seq": 65536}
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tpl, jobs, "test-model", defaults, GlmParser))
threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Thread(target=fake_engine, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}"


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


W = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
     "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "opts": {"type": "object"}}}}}
HIST = [{"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "reasoning_content": "need it",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
CALL = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value><arg_key>days</arg_key><arg_value>2</arg_value>"
        "<arg_key>opts</arg_key><arg_value>{\"units\": \"C\", \"hourly\": [1, 2.5]}</arg_value></tool_call>")
REQ = {
    "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=abc;"},
               {"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}],
    "tools": [{"name": "Bash", "description": "Run a command",
               "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}},
              {"type": "web_search_20250305", "name": "web_search"}],
    "messages": [{"role": "user", "content": "List the files."}],
    "max_tokens": 512, "thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
}
BASH = ("I should list files.</think>\nLet me look.<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call><|observation|>")
U = [{"role": "user", "content": "Hi"}]

cases = []


def C(name, path, body, ans, follow=None, big=None):
    cases.append({"name": name, "path": path, "body": body, "answer": ans, **({"follow": follow} if follow else {}),
                  **({"big": big} if big else {})})


C("chat: a tool call, max_tokens clamped", "/v1/chat/completions", {"messages": HIST, "tools": [W], "max_tokens": 100000},
  "ok</think>Checking." + CALL + "<|observation|>")
C("chat: content and reasoning", "/v1/chat/completions", {"messages": U}, "Let me think.</think>Héllo wörld!<|user|>")
C("chat: max_tokens inside reasoning", "/v1/chat/completions", {"messages": U, "max_tokens": 3}, "a long line of reasoning</think>x")
C("chat: reasoning only, then a stop", "/v1/chat/completions", {"messages": U, "max_completion_tokens": 50}, "only thinking<|user|>")
C("chat: effort from reasoning.effort", "/v1/chat/completions", {"messages": U, "reasoning": {"effort": "high"}, "reasoning_effort": "low",
                                                               "chat_template_kwargs": {"reasoning_effort": "low"}}, "x</think>y<|endoftext|>")
C("chat: effort from chat_template_kwargs", "/v1/chat/completions", {"messages": U, "chat_template_kwargs": {"reasoning_effort": "high"},
                                                                    "temperature": 0.6, "top_p": 0.9, "seed": 42}, "x</think>y<|endoftext|>")
C("chat: two calls and text after", "/v1/chat/completions", {"messages": U, "tools": [W]},
  "r</think>A" + CALL + "\nB" + CALL.replace("Rome", "Oslo") + "C<|observation|>")
C("chat stream: a tool call", "/v1/chat/completions", {"messages": HIST, "tools": [W], "stream": True},
  "ok</think>Checking." + CALL + "<|observation|>")
C("chat stream: unicode split across ids", "/v1/chat/completions", {"messages": U, "stream": True},
  "思考 🧑‍🔬</think>\n\n𠜎 ﷺ é́ done<|user|>")
C("chat stream: cut at max_tokens", "/v1/chat/completions", {"messages": U, "stream": True, "max_tokens": 4}, "abc def ghi</think>never")
C("completions", "/v1/completions", {"prompt": "Once upon a time", "max_tokens": 20}, "A reply.<|endoftext|>more")
C("debug prompt", "/v1/debug/prompt", {"messages": HIST, "tools": [W], "reasoning_effort": "high"}, "")
C("messages: a tool use", "/v1/messages", REQ, BASH)
C("messages: the follow-up restores the reasoning", "/v1/messages", None, "Done.</think>\n\nThe files are listed.<|user|>", follow=True)
C("messages stream: a tool use", "/v1/messages?beta=true", dict(REQ, stream=True), BASH)
C("messages stream: text after whitespace", "/v1/messages", dict(REQ, stream=True), "Thinking.</think>\n\n  Hello there!<|user|>")
C("messages: max_tokens", "/v1/messages", dict(REQ, max_tokens=3), "a b c d e</think>never")
C("messages: answer by text", "/v1/messages", dict(REQ, messages=[{"role": "user", "content": "Say it."}]), "R.</think>Sure thing.<|user|>")
C("messages: its reasoning comes back by text", "/v1/messages", dict(REQ, messages=[
    {"role": "user", "content": "Say it."}, {"role": "assistant", "content": [{"type": "text", "text": "Sure thing."}]},
    {"role": "user", "content": [{"type": "text", "text": "Again."}, {"type": "image", "source": {"type": "base64", "data": "QUJD"}}]},
    {"role": "system", "content": "Be brief."}]), "ok</think>Sure.<|user|>")
C("messages: count_tokens", "/v1/messages/count_tokens", REQ, "")
C("messages: too long", "/v1/messages", None, "", big=70000)

prev = None
for c in cases:
    if c.get("follow"):
        msg = json.loads(prev)
        use = [b for b in msg["content"] if b["type"] == "tool_use"][0]
        c["body"] = dict(REQ, messages=REQ["messages"] + [
            {"role": "assistant", "content": msg["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "a.txt"}]}])
        body = c["body"]
        del c["body"]           # built from the previous reply on each side
    elif c.get("big"):
        body = dict(REQ, messages=[{"role": "user", "content": "word " * c["big"]}])
        del c["body"]           # built on each side: "word " big times
    else:
        body = c["body"]
    answer["ids"] = tpl.encode(c["answer"])
    before = len(seen)
    c["status"], c["content_type"], c["response"] = post(c["path"], copy.deepcopy(body))
    if len(seen) > before:
        c["prompt"], c["max_tokens"] = seen[-1]
    prev = c["response"]

sys.stdout.write('{"cases": [\n' + ",\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n]}\n")
print(len(cases), "server cases", file=sys.stderr)
for c in cases:
    print("  ", c["name"], c["status"], c["content_type"], c["response"][:100].replace("\n", " "), file=sys.stderr)
