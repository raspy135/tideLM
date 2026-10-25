"""The GLM-5.3 fixture: server/glm_app.py's own GlmTemplate (the checkpoint's chat_template.jinja
under jinja2) and GlmParser, over a varied case set. The Rust test (glm.rs) renders, tokenizes and
parses the same cases and compares byte for byte, ids included.

    docker run --rm --user $(id -u):$(id -g) -v /home/ryan/git/tidelm:/w:ro \
        -v /home/ryan/models:/models:ro tidelm:local \
        python3 /w/rust/crates/tidelm-text/tests/gen_glm_cases.py /models/GLM-5.3-EXL3-3.0bpw \
        > rust/crates/tidelm-text/tests/glm_cases.json

(any Python with jinja2 + tokenizers; the repo root must be importable: server.glm_app is
imported, not copied)."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from server.glm_app import GlmParser, GlmTemplate  # noqa: E402

tpl = GlmTemplate(sys.argv[1])

# ------------------------------------------------------------------------------------- tools
W = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}}
S = {"type": "function", "function": {"name": "search", "description": "Search the \"web\"\nfor pages – ünïcode ✓",
     "parameters": {"properties": {"q": {"type": "string"}, "limit": {"type": "number", "default": 1.5e10},
                                   "frac": {"type": "number", "default": 0.1}, "tiny": {"default": 1e-7},
                                   "whole": {"default": 100.0}, "flag": {"type": "boolean", "default": False},
                                   "none": {"default": None}, "tags": {"type": "array", "items": {"type": "string"}}},
                    "type": "object"}, "strict": True}}
D = {"type": "function", "function": {"name": "deferred", "description": "never listed", "defer_loading": True,
     "parameters": {"type": "object", "properties": {}}}}
DF = {"type": "function", "function": {"name": "not_deferred", "defer_loading": False, "parameters": {}}}
OUTER = {"type": "function", "defer_loading": True, "function": {"name": "outer_defer", "parameters": {}}}
FLAT = {"name": "flat_tool", "description": "no function wrapper", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}, "obj": {"type": "object"}, "n": {"type": ["integer", "null"]}}}}
TOOLS = [W, S]


def call(id_, name, args, **kw):
    c = {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}
    c.update(kw)
    return c


IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

render_cases = []


def R(name, messages, tools=None, effort=None):
    render_cases.append({"name": name, "messages": messages, "tools": tools, "effort": effort})


# the spec's worked example (spec_glm_serving.md section 5)
worked = [{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "Weather?"},
          {"role": "assistant", "content": "Checking.", "reasoning_content": "think1",
           "tool_calls": [call("c1", "get_weather", json.dumps({"city": "Paris", "days": 3})),
                          call("c2", "get_weather", json.dumps({"city": "Rome"}))]},
          {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
          {"role": "tool", "tool_call_id": "c2", "content": "rainy"},
          {"role": "assistant", "content": "Done."},
          {"role": "user", "content": [{"type": "text", "text": "Thanks"}, IMG]}]
R("worked example, low", worked, [W], "low")
R("worked example, no effort", worked, [W])
R("plain system and user", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
for e in ["low", "high", "medium", "max", "LOW", "", 5, "none", "xhigh"]:
    R(f"effort {e!r}", [{"role": "user", "content": "Hi"}], None, e)
R("no messages", [])
R("empty contents", [{"role": "system", "content": ""}, {"role": "user", "content": ""},
                     {"role": "assistant", "content": ""}, {"role": "user", "content": None},
                     {"role": "user"}])
R("inline think in content", [{"role": "user", "content": "a"},
                              {"role": "assistant", "content": "<think>I reason.</think>\n\n  The answer.  \n"},
                              {"role": "user", "content": "b"}])
R("think end without start, several", [{"role": "user", "content": "a"},
                                       {"role": "assistant", "content": "r1</think>mid</think>last"},
                                       {"role": "assistant", "content": "x<think>y</think>z"}])
R("reasoning_content wins over inline", [{"role": "user", "content": "a"},
                                         {"role": "assistant", "content": "<think>inline</think>shown",
                                          "reasoning_content": "kept"}])
R("reasoning_content empty and null", [{"role": "user", "content": "a"},
                                       {"role": "assistant", "content": "x", "reasoning_content": ""},
                                       {"role": "assistant", "content": "y", "reasoning_content": None},
                                       {"role": "assistant", "content": "p</think>q", "reasoning_content": None}])
R("reasoning does not leak to the next turn", [{"role": "user", "content": "a"},
                                               {"role": "assistant", "content": "one", "reasoning_content": "R1"},
                                               {"role": "user", "content": "b"},
                                               {"role": "assistant", "content": "two"},
                                               {"role": "assistant", "content": "i</think>three"},
                                               {"role": "assistant", "content": "four"}])
R("whitespace-only content", [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": " \n\t ", "reasoning_content": "r"},
                              {"role": "assistant", "content": " x \x1c"}])
R("tool call arguments shapes", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": None, "tool_calls": [
        call("1", "f", '{"s": "str", "i": 3, "neg": -7, "f": 1.50, "big": 1e10, "small": -0.00001, "t": true, '
                       '"n": null, "arr": [1, "two", [3.0], {"k": "v"}], "obj": {"z": 1, "a": [true, null]}, '
                       '"u": "ü\\n\\"q\\" ✓ 日本", "e": "", "zero": 0.0, "negzero": -0.0, "exp": 2.5e-8, '
                       '"intzero": -0, "s-0": "-0", "list": [-0, -0.5, -0e1]}'),
        call("2", "f", "not json"),
        call("3", "f", ""),
        call("4", "f", "{}"),
        call("5", "f", {"already": "an object", "n": 2, "nested": {"x": [1]}}),
        call("6", "f", '  {"spaced": 1}  '),
        call("7", "f", '{"dup": 1, "other": 2, "dup": 3}'),
        call("8", "f", '{"html": "<b>&amp;</b>", "tag": "</arg_value>", "cjk": "中文", "emoji": "🧑‍🔬"}')]},
    {"role": "tool", "tool_call_id": "1", "content": "ok"}], [W])
R("tool call without the function wrapper", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "name": "flat",
                                                          "arguments": {"a": 1, "b": "two"}}]}])
R("empty tool_calls list", [{"role": "user", "content": "go"},
                            {"role": "assistant", "content": "plain", "tool_calls": []}])
R("tool results reordered by call id", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "reasoning_content": "two calls",
     "tool_calls": [call("a", "get_weather", '{"city": "A"}'), call("b", "get_weather", '{"city": "B"}'),
                    call("c", "get_weather", '{"city": "C"}')]},
    {"role": "tool", "tool_call_id": "c", "content": "rc"},
    {"role": "tool", "tool_call_id": "a", "content": "ra"},
    {"role": "tool", "tool_call_id": "b", "content": "rb"},
    {"role": "user", "content": "next"}], [W])
R("tool results keyed by id (not tool_call_id)", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}")]},
    {"role": "tool", "id": "b", "content": "rb"},
    {"role": "tool", "id": "a", "content": "ra"}])
R("tool results: a missing id keeps the order", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}")]},
    {"role": "tool", "tool_call_id": "b", "content": "rb"},
    {"role": "tool", "content": "no id"}])
R("tool results: an unknown id keeps the order", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}")]},
    {"role": "tool", "tool_call_id": "b", "content": "rb"},
    {"role": "tool", "tool_call_id": "zzz", "content": "rz"}])
R("tool results: a duplicate result id keeps the order", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}")]},
    {"role": "tool", "tool_call_id": "b", "content": "rb1"},
    {"role": "tool", "tool_call_id": "b", "content": "rb2"},
    {"role": "tool", "tool_call_id": "a", "content": "ra"}])
R("tool results: duplicate call ids keep the order", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("a", "f", "{}")]},
    {"role": "tool", "tool_call_id": "a", "content": "r2"},
    {"role": "tool", "tool_call_id": "a", "content": "r1"}])
R("tool results: fewer results than calls", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}"), call("c", "f", "{}")]},
    {"role": "tool", "tool_call_id": "c", "content": "rc"},
    {"role": "tool", "tool_call_id": "a", "content": "ra"}])
R("tool messages without an assistant before", [
    {"role": "tool", "tool_call_id": "a", "content": "first"},
    {"role": "user", "content": "u"},
    {"role": "tool", "tool_call_id": "b", "content": "after user"},
    {"role": "tool", "tool_call_id": "c", "content": "second"}])
R("tool contents: parts, null, media", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}"), call("c", "f", "{}")]},
    {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}, "raw"]},
    {"role": "tool", "tool_call_id": "b", "content": None},
    {"role": "tool", "tool_call_id": "c", "content": [{"type": "image"}, {"type": "text", "text": "after"}]}])
R("tool_reference content", [
    {"role": "user", "content": "find tools"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "tool_search", '{"q": "weather"}')]},
    {"role": "tool", "tool_call_id": "a", "content": [{"type": "tool_reference", "name": "get_weather"},
                                                     {"type": "tool_reference", "name": "deferred"},
                                                     {"type": "tool_reference", "name": "missing"}]}],
  [W, D, S])
R("list of outputs, sorted", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}"), call("c", "f", "{}")]},
    {"role": "tool", "content": [{"tool_call_id": "c", "output": "out c"},
                                 {"id": "a", "output": [{"type": "text", "text": "out a"}, IMG]}]},
    {"role": "tool", "content": [{"id": "b", "output": [{"type": "tool_reference", "name": "search"}]}]}],
  [W, S])
R("list of outputs, unsorted (no assistant calls)", [
    {"role": "user", "content": "go"},
    {"role": "tool", "content": [{"id": "c", "output": "out c"}, {"id": "a", "output": None},
                                 {"output": [{"type": "tool_reference", "name": "get_weather"}]},
                                 {"id": "z"}]}],
  [W])
R("list of outputs, an id missing", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("a", "f", "{}"), call("b", "f", "{}")]},
    {"role": "tool", "content": [{"id": "b", "output": "ob"}, {"output": "no id"}]}])
R("media parts and odd items", [
    {"role": "system", "content": [{"type": "text", "text": "sys1"}, {"type": "text", "text": "sys2"}]},
    {"role": "user", "content": [{"type": "text", "text": "look:"}, {"type": "image"}, {"type": "image_url"},
                                 {"type": "video"}, {"type": "video_url"}, {"type": "audio"}, {"type": "audio_url"},
                                 {"type": "input_audio"}, {"type": "file"}, {"type": "text"}, {"type": "text", "text": None},
                                 {"type": "text", "text": 42}, "plain string", 7, ["nested"], {"no": "type"},
                                 {"type": "input_image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "a1"}, {"type": "image"}, {"type": "text", "text": " a2 "}]}])
R("unicode and typed specials", [
    {"role": "system", "content": "日本語のシステム 🧑‍🔬 𠜎 ﷺ é́"},
    {"role": "user", "content": "say <|user|> and </think> and <tool_call>   \x00 tab\tend"},
    {"role": "assistant", "content": "ok 😀", "reasoning_content": "думаю…"}])
R("other roles render nothing", [
    {"role": "developer", "content": "dev note"}, {"role": "user", "content": "u"},
    {"role": "function", "content": "f"}, {"content": "no role"}, {"role": None, "content": "null role"},
    {"role": "system", "content": "mid system"}, {"role": "latest_reminder", "content": "x"}])
R("tools: defer_loading, strict, flat, outer", [{"role": "user", "content": "u"}], [W, S, D, DF, OUTER, FLAT])
R("tools: only deferred", [{"role": "user", "content": "u"}], [D])
R("tools: empty list", [{"role": "user", "content": "u"}], [])
R("tools with a tool history and a system", [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "read it"},
    {"role": "assistant", "content": "Reading.", "reasoning_content": "need the file",
     "tool_calls": [call("t1", "flat_tool", '{"path": "/tmp/a b.txt", "obj": {"k": [1, 2]}, "n": null}')]},
    {"role": "tool", "tool_call_id": "t1", "content": "line1\nline2\n"},
    {"role": "assistant", "content": "It has two lines."},
    {"role": "user", "content": "thanks"}], [FLAT], "high")
R("long multi-turn agent", [
    {"role": "system", "content": "s"}] + sum([[
        {"role": "user", "content": f"step {i}"},
        {"role": "assistant", "content": f"doing {i}", "reasoning_content": f"plan {i}",
         "tool_calls": [call(f"id{i}a", "get_weather", json.dumps({"city": f"C{i}", "days": i})),
                        call(f"id{i}b", "search", json.dumps({"q": f"q{i}", "tags": ["x", i]}))]},
        {"role": "tool", "tool_call_id": f"id{i}b", "content": f"res b{i}"},
        {"role": "tool", "tool_call_id": f"id{i}a", "content": f"res a{i}"}] for i in range(3)], []),
  TOOLS, "low")
R("non-string contents are str()'d", [
    {"role": "system", "content": 3}, {"role": "user", "content": True}, {"role": "user", "content": 1.5},
    {"role": "user", "content": {"type": "text", "text": "it's é", "n": [1, 2.5, None, True, False], "q": "say \"hi\"",
                                 "both": "'\"", "ctl": "a\tb\x01 ​\U0001F600"}},
    {"role": "assistant", "content": [{"type": "text", "text": "r</think>"}, {"type": "text", "text": " visible "}]}])
R("falsy tool contents", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("1", "f", "{}"), call("2", "f", "{}"), call("3", "f", "{}"),
                                                         call("4", "f", "{}")]},
    {"role": "tool", "tool_call_id": "1", "content": 0}, {"role": "tool", "tool_call_id": "2", "content": False},
    {"role": "tool", "tool_call_id": "3", "content": {}}, {"role": "tool", "tool_call_id": "4", "content": []}])
R("numeric ids compare as rendered text", [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [call("1", "f", "{}"), call("2", "f", "{}")]},
    {"role": "tool", "tool_call_id": 2, "content": "two"}, {"role": "tool", "tool_call_id": 1, "content": "one"}])
R("a nameless reference matches a nameless tool", [
    {"role": "user", "content": "go"},
    {"role": "tool", "content": [{"type": "tool_reference"}]}],
  [W, {"type": "function", "function": {"description": "no name", "parameters": {}}}])
# errors: jinja2 or glm_app raise; the Rust renderer must refuse too
R("err arguments are a list", [{"role": "user", "content": "x"},
                               {"role": "assistant", "content": "", "tool_calls": [call("1", "f", "[1, 2]")]}])
R("err arguments are a string", [{"role": "user", "content": "x"},
                                 {"role": "assistant", "content": "", "tool_calls": [call("1", "f", '"s"')]}])
R("err arguments null", [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": "", "tool_calls": [call("1", "f", None)]}])
R("err call without a name", [{"role": "user", "content": "x"},
                              {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"arguments": "{}"}}]}])
R("err tool content is an object", [{"role": "user", "content": "x"},
                                    {"role": "tool", "tool_call_id": "1", "content": {"a": 1}}])
R("err tool_reference without tools", [{"role": "user", "content": "x"},
                                       {"role": "tool", "content": [{"type": "tool_reference", "name": "f"}]}])

for c in render_cases:
    try:
        text = tpl.render(copy.deepcopy(c["messages"]), copy.deepcopy(c["tools"]), c["effort"])
        c["text"], c["ids"] = text, tpl.encode(text)
    except Exception as e:  # noqa: BLE001 - any refusal is a refusal
        c["error"] = f"{type(e).__name__}: {e}"
    if c["name"].startswith("err") != ("error" in c):
        raise SystemExit(f"case {c['name']}: {c.get('error')}")

# ------------------------------------------------------------------------------------- parser
parse_cases = []


def P(name, text=None, ids=None, tools=None):
    parse_cases.append({"name": name, "ids": ids if ids is not None else tpl.encode(text), "tools": tools,
                        **({"source": text} if text is not None else {})})


ARG = "<arg_key>{}</arg_key><arg_value>{}</arg_value>"
P("reasoning, content, stop", "Let me think.</think>Héllo wörld!<|user|>")
P("reasoning only (cut off)", "Still thinking about 𠜎 and 🧑‍🔬")
P("tool call round trip", "ok</think>Checking.<tool_call>get_weather" + ARG.format("city", "Paris")
  + ARG.format("days", "3") + "</tool_call><|observation|>", tools=[W])
P("two calls, text between and after", "plan</think>\nFirst.<tool_call>get_weather\n" + ARG.format("city", "Oslo")
  + "</tool_call>\nthen<tool_call>search" + ARG.format("q", "rust 1.0") + ARG.format("limit", "5")
  + ARG.format("tags", '["a", "b"]') + "</tool_call>tail<|observation|>", tools=TOOLS)
P("argument shapes, no schema", "</think><tool_call>  any_tool  " + ARG.format(" spaced key ", "3")
  + "\n" + ARG.format("obj", '{"z": 1, "a": [1.5, 2e-9, null, true]}') + ARG.format("bad", "{not json")
  + ARG.format("word", "Paris") + ARG.format("str", '"quoted"') + ARG.format("float", "1.0")
  + ARG.format("exp", "1e5") + ARG.format("negzero", "-0") + ARG.format("nz_list", '[-0, "-0", -0.0, {"-0": -0}]')
  + ARG.format("ws", " 7 ")
  + ARG.format("multi", "line1\nline2") + ARG.format("uni", '"日本 ✓"') + ARG.format("uni_raw", "日本 ✓")
  + ARG.format("dup", "1") + ARG.format("dup", "2") + ARG.format("empty", "") + "</tool_call>")
P("argument shapes, schema types", "</think><tool_call>flat_tool" + ARG.format("path", "123")
  + ARG.format("obj", '{"k": "v"}') + ARG.format("n", "null") + ARG.format("extra", "[1,2]") + "</tool_call>",
  tools=[FLAT, {"type": "function", "function": {"name": "flat_tool", "parameters": {"properties": {"path": {"type": "string"}}}}}])
P("whitespace between key and value", "</think><tool_call>f<arg_key>a</arg_key>\n  \t<arg_value>1</arg_value>"
  "<arg_key>b</arg_key>x<arg_value>2</arg_value><arg_key>c</arg_key><arg_value>3</arg_value></tool_call>")
P("no arguments", "</think><tool_call>noop</tool_call><tool_call>\n</tool_call>")
P("unterminated call", "</think>calling<tool_call>get_weather" + ARG.format("city", "Lima") + "<arg_key>days", tools=[W])
P("think again after think end", "a</think>b<think>c</think>d")
P("stop ids are dropped mid-stream", "r<|endoftext|>s</think>t<|user|>u<|observation|>v")
P("specials inside a call", "</think><tool_call>f<think>" + ARG.format("k", "v</think>") + "<|user|></tool_call>after")
P("unicode split across tokens", "</think>" + "𠜎🧑‍🔬ﷺ é́ 日本語" * 2)
P("empty", "")
# hand-made ids: partial UTF-8 bytes (ByteLevel symbols of single bytes)
bl = {b: chr(c) for b, c in zip(
    list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)),
    list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)))}
n = 0
for b in range(256):
    if b not in bl:
        bl[b] = chr(256 + n)
        n += 1


def byte_id(b):
    i = tpl.tok.token_to_id(bl[b])
    assert i is not None, b
    return i


A = tpl.encode("A")
P("partial bytes then ascii", ids=[byte_id(0xE2), byte_id(0x82)] + A + [byte_id(0xE2), byte_id(0x82), byte_id(0xAC)] + A)
P("a partial character before a switch", ids=[byte_id(0xF0), byte_id(0x9F), tpl.think_end, byte_id(0xE6)])
P("invalid byte inside", ids=[byte_id(0xC3)] + A + [byte_id(0xFF)] + A + [byte_id(0xE2)])

for c in parse_cases:
    p = GlmParser(tpl)
    ev = []
    for t in c["ids"]:
        ev += [list(e) for e in p.feed(t)]
    fin = [list(e) for e in p.finish()]
    calls = p.tool_calls(copy.deepcopy(c["tools"]))
    c["events"], c["final"] = ev, fin
    c["reasoning"], c["content"] = p.texts["reasoning"], p.texts["content"]
    c["call_texts"] = [tpl.decode(x) for x in p.calls]
    # ids are call_<random>: everything else
    c["tool_calls"] = [[x["function"]["name"], x["function"]["arguments"]] for x in calls]

out = {"stop": sorted(tpl.stop), "think": tpl.think, "think_end": tpl.think_end, "call": tpl.call,
       "call_end": tpl.call_end, "render": render_cases, "parse": parse_cases}
sys.stdout.write("{\n")
for i, k in enumerate(out):
    v = out[k]
    if isinstance(v, list) and v and isinstance(v[0], dict):
        sys.stdout.write(f' "{k}": [\n' + ",\n".join("  " + json.dumps(x) for x in v) + "\n ]")
    else:
        sys.stdout.write(f' "{k}": {json.dumps(v)}')
    sys.stdout.write(",\n" if i < len(out) - 1 else "\n")
sys.stdout.write("}\n")
print(len(render_cases), "render cases,", len(parse_cases), "parse cases", file=sys.stderr)
for c in render_cases:
    if "error" in c:
        print("  ", c["name"], "->", c["error"], file=sys.stderr)
