"""python3 gen_chat_tools_cases.py chat_tools_cases.json: the fixture, from the checkpoint's encoder
(encode_messages, parse_message_from_completion_text) and server/app.py's tolerant parser."""
import copy, json, re, sys
sys.path.insert(0, "/home/ryan/models/DeepSeek-V4.1-Flash-1017/encoding")
import encoding as enc

# server/app.py's _parse_tool_calls_tolerant, verbatim
_RE_INVOKE = re.compile(r'<｜DSML｜ invoke name="(?P<name>[^"]*)"\s*>?\n?(?P<body>.*?)(?=<｜DSML｜ invoke |</｜DSML｜ calls>|\Z)', re.DOTALL)
_RE_PARAM_SPEC = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<s>true|false)"\s*>(?P<v>.*?)<', re.DOTALL)
_RE_PARAM_ATTR = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<v>.*?)"\s*>', re.DOTALL)
def tolerant(text):
    out = []
    for m in _RE_INVOKE.finditer(text):
        name, body = m.group("name"), m.group("body")
        if not name:
            continue
        args = {}
        for pm in _RE_PARAM_SPEC.finditer(body):
            k, is_str, v = pm.group("k"), pm.group("s"), pm.group("v")
            if is_str == "true":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except Exception:
                    args[k] = v
        consumed = {pm.group("k") for pm in _RE_PARAM_SPEC.finditer(body)}
        for pm in _RE_PARAM_ATTR.finditer(body):
            k, v = pm.group("k"), pm.group("v")
            if k and k not in consumed and k not in args:
                args[k] = v
        out.append([name, json.dumps(args, ensure_ascii=False)])
    return out

W = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
     "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["location"]}}}
S = {"type": "function", "function": {"name": "search", "description": "Search the web",
     "parameters": {"properties": {"q": {"type": "string"}, "limit": {"type": "number", "default": 1.5e10}}, "type": "object"}},
     "namespace": {"name": "web", "description": "Web tools"}}
N = {"type": "function", "namespace": "fs", "function": {"name": "read", "parameters": {}}}
Q = {"type": "function", "function": {"name": "fs::write"}}

def sys_tools(tools, content=""):
    return {"role": "system", "content": content, "tools": tools}

def call(id_, name, args, **kw):
    c = {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}
    c.update(kw)
    return c

worked = [sys_tools([W]),
          {"role": "user", "content": "Weather in Berlin?"},
          {"role": "assistant", "content": "Let me look that up.", "reasoning_content": "The user wants weather.",
           "tool_calls": [call("call_1", "get_weather", '{"location":"Berlin","days":3}')]},
          {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'}]

enc_cases = []
def E(name, messages, thinking=False, effort=75):
    enc_cases.append({"name": name, "messages": messages, "thinking": thinking, "effort": effort})

E("worked chat", worked)
E("worked thinking", worked, True)
E("worked thinking effort 60, answer after", worked + [{"role": "assistant", "content": "It is 21 C."},
                                                       {"role": "user", "content": "And tomorrow?"}], True, 60)
E("namespaces", [sys_tools([W, S, N, Q], "You are helpful."), {"role": "user", "content": "go"},
                 {"role": "assistant", "content": None, "tool_calls": [
                     call("a", "web::search", '{"q": "rust", "limit": 3}'),
                     call("b", "read", {"path": "/tmp/x"}, namespace="fs"),
                     call("c", "write", '{"text": "hi"}', namespace="fs")]},
                 {"role": "tool", "tool_call_id": "c", "content": "ok"},
                 {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "r1"}, {"type": "image_url", "image_url": "u"},
                                                                     {"type": "file"}]},
                 {"role": "tool", "tool_call_id": "b", "content": None},
                 {"role": "user", "content": "thanks"}], True)
E("arguments shapes", [sys_tools([W]), {"role": "user", "content": "x"},
                       {"role": "assistant", "content": "", "tool_calls": [
                           call("1", "f", json.dumps(json.dumps({"a": 1}))),
                           call("2", "f", "not json"),
                           call("3", "f", "[1,2]"),
                           call("4", "f", '{"n": 1.50, "big": 1e10, "i": 3, "neg": -0.00001, "u": "ü\\n\\"q\\"", "o": {"z": 1, "a": [true, null]}}'),
                           call("5", "f", "{}"),
                           call("6", "f", '"just a string"'),
                           call("7", "f", "12")]},
                       {"role": "tool", "content": "done"}], False)
E("tool history without tools, thinking", [{"role": "user", "content": "q"},
     {"role": "assistant", "content": "", "reasoning_content": "r0", "tool_calls": [call("1", "f", "{}")]},
     {"role": "tool", "tool_call_id": "1", "content": "res"}], True)
E("tool history without tools, then user, thinking", [{"role": "user", "content": "q"},
     {"role": "assistant", "content": "", "reasoning_content": "r0", "tool_calls": [call("1", "f", "{}")]},
     {"role": "tool", "tool_call_id": "1", "content": "res"},
     {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
     {"role": "user", "content": "next"}], True)
E("mid system", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                 {"role": "system", "content": "be brief"}], True)
E("response format", [{"role": "system", "content": "s", "response_format": {"type": "object", "properties": {"x": {"type": "number"}}}},
                      {"role": "user", "content": "u"}])
E("list contents", [{"role": "system", "content": [{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}]},
                    {"role": "user", "content": [{"type": "text", "text": "u1"}, {"type": "video"}, {"type": "text", "text": "u2"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "a1"}, {"type": "x"}]},
                    {"role": "user", "content": "u3"}, {"role": "user", "content": "u4"}])
E("latest reminder and wo_eos", [{"role": "user", "content": "u"}, {"role": "latest_reminder", "content": "remember"},
                                 {"role": "assistant", "content": "partial", "wo_eos": True}])
E("task query", [{"role": "user", "content": "find", "task": "query"}])
E("task action thinking", [{"role": "user", "content": "do", "task": "action"}, {"role": "assistant", "content": "act"}], True)
E("images", [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                          {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}},
                                          {"type": "image_url", "image_url": "http://x/y.png"}]}])
E("tool thinking keeps reasoning", [sys_tools([W]), {"role": "user", "content": "a"},
                                    {"role": "assistant", "content": "b", "reasoning_content": "rb"},
                                    {"role": "user", "content": "c"}], True)
E("err unknown role", [{"role": "wizard", "content": "x"}])
E("err call without function", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}])
E("err call without arguments", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]}])
E("err name with ::", [sys_tools([{"function": {"name": "a::b::c"}}]), {"role": "user", "content": "x"}])
E("err conflicting namespace", [sys_tools([{"namespace": "x", "function": {"name": "y::f"}}]), {"role": "user", "content": "x"}])
E("err tool without function", [sys_tools([{"type": "function"}]), {"role": "user", "content": "x"}])
E("err typed placeholder", [{"role": "user", "content": "see <｜deepseek_image｜>"}])
E("err placeholder in a text block", [{"role": "user", "content": [{"type": "text", "text": "<｜deepseek_image｜>"}]}])
E("err image without source", [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])
E("err bad task", [{"role": "user", "content": "x", "task": "nope"}])

for c in enc_cases:
    try:
        prompt, media = enc.encode_messages(copy.deepcopy(c["messages"]), thinking_mode="thinking" if c["thinking"] else "chat",
                                            reasoning_effort=c["effort"], return_multi_modal_data=True)
        c["prompt"], c["images"] = prompt, media["images"]
    except (AssertionError, ValueError, KeyError, TypeError, NotImplementedError) as e:
        c["error"] = str(e)
    if c["name"].startswith("err") != ("error" in c):
        raise SystemExit(f"case {c['name']}: {c.get('error')}")

B = "\n\n<｜DSML｜ calls>\n"
INV = '<｜DSML｜ invoke name="{}">\n'
P = '<｜DSML｜ parameter name="{}" string="{}">{}</｜DSML｜ parameter>\n'
END_INV = "</｜DSML｜ invoke>\n"
END = "</｜DSML｜ calls>"
EOS = enc.eos_token
parse_cases = []
def Pc(name, text, thinking=False):
    parse_cases.append({"name": name, "text": text, "thinking": thinking})

one = B + INV.format("get_weather") + P.format("location", "true", "Berlin") + END_INV + END
Pc("one call", "Let me look that up." + one + EOS)
Pc("one call thinking", "The user is talking about weather.</think>Let me look." + one + EOS, True)
Pc("two calls, typed", B + INV.format("a") + P.format("x", "false", "[1, 2]") + P.format("y", "true", "a \"q\"\nline") + END_INV
   + INV.format("web::b") + P.format("n", "false", "3") + END_INV + END + EOS)
Pc("zero params", B + '<｜DSML｜ invoke name="f">\n\n</｜DSML｜ invoke>\n' + END + EOS)
Pc("value then newline", B + INV.format("g") + '<｜DSML｜ parameter name="city" string="true">Berlin<\n/｜DSML｜ parameter>\n'
   + P.format("days", "false", "3") + END_INV + END + EOS)
Pc("raw invalid false value", B + INV.format("h") + P.format("x", "false", "abc") + END_INV + END + EOS)
Pc("duplicate param", B + INV.format("h") + P.format("x", "true", "1") + P.format("x", "true", "2") + END_INV + END + EOS)
Pc("prose after the block", "ok" + one + "\nDone." + EOS)
Pc("value in the attribute", B + INV.format("s") + '<｜DSML｜ parameter name="query" string="unified memory specs">\n' + END_INV + END + EOS)
Pc("value in the attribute, prose", B + INV.format("s") + '<｜DSML｜ parameter name="query" string="unified memory specs">\n' + END_INV + END + "\nSure." + EOS)
Pc("missing think end", "reasoning only" + EOS, True)
Pc("no calls", "plain answer" + EOS)
Pc("plain text, no eos", "plain answer")
Pc("cut off mid value", B + INV.format("w") + '<｜DSML｜ parameter name="x" string="true">Berl' + EOS)
Pc("cut off after >", B + INV.format("w") + '<｜DSML｜ parameter name="x" string="true">' + EOS)
Pc("html value", B + INV.format("w") + P.format("html", "true", "<b>x</b>") + END_INV + END + EOS)
Pc("dsml in content", "a ｜DSML｜ b" + EOS)
Pc("bad separator", B.rstrip("\n") + "\n\n" + INV.format("a") + END_INV + END + EOS)
Pc("call mentioned in reasoning", 'think about <｜DSML｜ invoke name="g">\n</think>ok' + one + EOS, True)
Pc("empty block", B + END + EOS)

for c in parse_cases:
    try:
        r = enc.parse_message_from_completion_text(c["text"], thinking_mode="thinking" if c["thinking"] else "chat")
        c["parsed"] = {"content": r["content"], "reasoning": r["reasoning_content"],
                       "tool_calls": [[t["function"]["name"], t.get("namespace"), t["function"]["arguments"]] for t in r["tool_calls"]]}
    except Exception as e:
        c["error"] = str(e)
    c["tolerant"] = tolerant(c["text"])

out = sys.argv[1]
json.dump({"encode": enc_cases, "parse": parse_cases}, open(out, "w"), ensure_ascii=False, indent=1)
print(len(enc_cases), "encode cases,", len(parse_cases), "parse cases ->", out)
for c in enc_cases:
    if "error" in c:
        print("  ", c["name"], "->", c["error"])
for c in parse_cases:
    print("  ", c["name"], "->", c.get("error") or c["parsed"]["tool_calls"], "| tolerant", c["tolerant"])
