//! The DeepSeek-V4.1 chat prompt and the parser of its answers: the checkpoint's own
//! `encoding/encoding.py` (the file server/app.py loads), on OpenAI-shaped JSON messages.
//! encode_messages renders system (leading and mid-conversation), user, assistant, tool and
//! latest_reminder turns, tool definitions and earlier calls in DSML, response-format schemas
//! and tasks, in chat or thinking mode with the numeric reasoning effort; image blocks become
//! `<｜deepseek_image｜>` placeholders, returned in prompt order for the caller to expand.
//! parse_completion is parse_message_from_completion_text, and parse_tool_calls_tolerant
//! server/app.py's fallback for a block the strict parser refuses.
//!
//! JSON goes into the prompt as Python's `json.dumps(ensure_ascii=False)` writes it - ", " and
//! ": " separators, keys in the order the client sent them, floats as repr - because those
//! bytes are the prompt (serde_json's preserve_order keeps the order).

use serde_json::{json, Map, Value};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const IMAGE: &str = "<｜deepseek_image｜>";
/// what opens a tool-calls block in an answer
pub const TOOL_CALLS_START: &str = "\n\n<｜DSML｜ calls";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const SYSTEM: &str = "<｜System｜>";
const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
const TASKS: [(&str, &str); 6] = [("action", "<｜action｜>"), ("query", "<｜query｜>"), ("authority", "<｜authority｜>"),
                                  ("domain", "<｜domain｜>"), ("title", "<｜title｜>"), ("read_url", "<｜read_url｜>")];
const CALLS_END: &str = "</｜DSML｜ calls>";
const INVOKE_START: &str = "<｜DSML｜ invoke";
const INVOKE_END: &str = "</｜DSML｜ invoke";
const PARAM_START: &str = "<｜DSML｜ parameter";
const PARAM_END: &str = "/｜DSML｜ parameter";
const RESPONSE_FORMAT: &str = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

const TOOLS_HEAD: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#;
const TOOLS_TAIL: &str = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

type Msg = Map<String, Value>;

// ---------------------------------------------------------------- Python's values

/// `json.dumps(v, ensure_ascii=False)`
pub fn py_json(v: &Value) -> String {
    let mut s = String::new();
    write_py(v, &mut s);
    s
}

fn write_py(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_number(n)),
        // the escapes json.dumps makes without ensure_ascii are serde_json's: '"', '\\' and
        // the control characters, the short forms where there are some
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap()),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                write_py(x, out);
            }
            out.push('}');
        }
    }
}

fn py_number(n: &serde_json::Number) -> String {
    match n.as_f64() {
        Some(f) if !(n.is_i64() || n.is_u64()) => py_float(f),
        _ => n.to_string(),
    }
}

/// `float.__repr__`: the shortest digits that read back, positional for decimal exponents in
/// [-4, 16), else d.ddde+XX
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let e = format!("{f:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut s = String::from(if neg { "-" } else { "" });
    if (-4..16).contains(&exp) {
        if exp < 0 {
            s.push_str("0.");
            s.push_str(&"0".repeat((-exp - 1) as usize));
            s.push_str(&digits);
        } else {
            let int = exp as usize + 1;
            if digits.len() <= int {
                s.push_str(&digits);
                s.push_str(&"0".repeat(int - digits.len()));
                s.push_str(".0");
            } else {
                s.push_str(&digits[..int]);
                s.push('.');
                s.push_str(&digits[int..]);
            }
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    }
    s
}

/// `str(v)`, as `"{content}".format(content=v)` writes it
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => py_number(n),
        other => py_json(other),
    }
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `v or ""`
fn or_empty(v: Option<&Value>) -> String {
    if truthy(v) { py_str(v.unwrap()) } else { String::new() }
}

/// `dict.get(k)`: a null value is None too
fn get<'a>(m: &'a Msg, k: &str) -> Option<&'a Value> {
    m.get(k).filter(|v| !v.is_null())
}

fn role(m: &Msg) -> Option<&str> {
    m.get("role").and_then(|r| r.as_str())
}

/// KeyError's message
fn key_error(k: &str) -> String {
    format!("'{k}'")
}

fn text_of(v: Option<&Value>) -> Result<String, String> {
    match v {
        None => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!("sequence item: expected str instance, {} found", type_name(other))),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn block_type(b: &Value) -> String {
    b.get("type").map(py_str).unwrap_or_else(|| "None".into())
}

// ---------------------------------------------------------------- tools

fn split_tool_name(name: &str, namespace: Option<&str>) -> Result<(Option<String>, String), String> {
    let (mut ns, mut name) = (namespace.map(String::from), name.to_string());
    if let Some((prefix, bare)) = name.clone().split_once("::") {
        if let Some(n) = &ns {
            if n != prefix {
                return Err(format!("Conflicting tool namespaces: {n} != {prefix}"));
            }
        }
        (ns, name) = (Some(prefix.into()), bare.into());
    }
    if name.contains("::") {
        return Err(format!("Tool name must not contain '::': {name}"));
    }
    if let Some(n) = ns.as_deref().filter(|n| n.contains("::")) {
        return Err(format!("Tool namespace must not contain '::': {n}"));
    }
    Ok((ns, name))
}

fn tool_name_for_encoding(tool: &Msg) -> Result<String, String> {
    let ns = match get(tool, "namespace") {
        None => None,
        Some(Value::Object(o)) => Some(o.get("name").map(py_str).ok_or_else(|| key_error("name"))?),
        Some(v) => Some(py_str(v)),
    };
    let name = tool.get("name").ok_or_else(|| key_error("name"))?;
    let (ns, name) = split_tool_name(&py_str(name), ns.as_deref())?;
    Ok(match ns {
        None => name,
        Some(n) => format!("{n}::{name}"),
    })
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Msg, String> {
    v.as_object().ok_or_else(|| format!("{what} must be an object"))
}

fn array<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>, String> {
    v.as_array().ok_or_else(|| format!("{what} must be a list"))
}

/// tools_from_openai_format: each tool's function, its name namespace-qualified and the
/// namespace's description ahead of its own
pub fn tools_from_openai_format(tools: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for tool in tools {
        let t = object(tool, "a tool")?;
        let mut f = object(t.get("function").ok_or_else(|| key_error("function"))?, "a tool's function")?.clone();
        if let Some(ns) = get(t, "namespace") {
            f.insert("namespace".into(), ns.clone());
        }
        let name = tool_name_for_encoding(&f)?;
        f.insert("name".into(), Value::String(name));
        if let Some(Value::Object(ns)) = f.shift_remove("namespace") {
            if truthy(ns.get("description")) {
                let desc = format!("{}\n{}", py_str(&ns["description"]), or_empty(f.get("description")));
                f.insert("description".into(), Value::String(desc));
            }
        }
        out.push(Value::Object(f));
    }
    Ok(out)
}

fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(py_json).collect();
    format!("{TOOLS_HEAD}{}{TOOLS_TAIL}", schemas.join("\n"))
}

/// tool_calls_from_openai_format: (name, arguments, namespace) a call
fn tool_calls_from_openai_format(calls: &[Value]) -> Result<Vec<Msg>, String> {
    let mut out = Vec::new();
    for tc in calls {
        let tc = object(tc, "a tool call")?;
        let f = object(tc.get("function").ok_or_else(|| key_error("function"))?, "a tool call's function")?;
        let ns = if truthy(tc.get("namespace")) { tc.get("namespace") } else { get(f, "namespace") }.map(py_str);
        let name = py_str(f.get("name").ok_or_else(|| key_error("name"))?);
        let (ns, name) = split_tool_name(&name, ns.as_deref())?;
        let mut call = Msg::new();
        call.insert("name".into(), Value::String(name));
        call.insert("arguments".into(), f.get("arguments").ok_or_else(|| key_error("arguments"))?.clone());
        if let Some(n) = ns {
            call.insert("namespace".into(), Value::String(n));
        }
        out.push(call);
    }
    Ok(out)
}

/// encode_arguments_to_dsml: a dict as it is, a JSON string (even encoded twice) decoded,
/// anything else as one "arguments" parameter
fn encode_arguments(call: &Msg) -> String {
    let orig = call.get("arguments").cloned().unwrap_or(Value::Null);
    let mut args = orig.clone();
    if !args.is_object() {
        for _ in 0..2 {
            match &args {
                Value::String(s) => match serde_json::from_str::<Value>(s) {
                    Ok(v) => args = v,
                    Err(_) => break,
                },
                _ => break,
            }
        }
        if !args.is_object() {
            args = json!({ "arguments": orig });
        }
    }
    let params: Vec<String> = args
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            let (is_str, value) = match v {
                Value::String(s) => ("true", s.clone()),
                other => ("false", py_json(other)),
            };
            format!("<{DSML} parameter name=\"{k}\" string=\"{is_str}\">{value}</{DSML} parameter>")
        })
        .collect();
    params.join("\n")
}

// ---------------------------------------------------------------- preprocessing

/// merge_tool_messages: a tool message becomes a tool_result block of a user turn, joined to
/// the user turn before it (as is a user message after a user turn without a task)
fn merge_tool_messages(messages: Vec<Msg>) -> Result<Vec<Msg>, String> {
    let mut merged: Vec<Msg> = Vec::new();
    let user_with_blocks = |m: Option<&Msg>| m.is_some_and(|m| role(m) == Some("user") && m.contains_key("content_blocks"));
    for mut msg in messages {
        match role(&msg) {
            Some("tool") => {
                let mut content = msg.get("content").cloned().unwrap_or(Value::String(String::new()));
                if truthy(msg.get("content_blocks")) && !msg.get("content").is_some_and(|c| c.is_string()) {
                    content = msg["content_blocks"].clone();
                }
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": content,
                });
                if user_with_blocks(merged.last()) {
                    blocks_mut(merged.last_mut().unwrap())?.push(block);
                } else {
                    let mut m = Msg::new();
                    m.insert("role".into(), "user".into());
                    m.insert("content_blocks".into(), Value::Array(vec![block]));
                    merged.push(m);
                }
            }
            Some("user") => {
                let blocks = match get(&msg, "content_blocks") {
                    Some(b) => b.clone(),
                    None => match msg.get("content").cloned().unwrap_or(Value::String(String::new())) {
                        Value::Array(a) => Value::Array(a),
                        c => json!([{ "type": "text", "text": c }]),
                    },
                };
                if user_with_blocks(merged.last()) && get(merged.last().unwrap(), "task").is_none() {
                    let more = array(&blocks, "content_blocks")?.clone();
                    blocks_mut(merged.last_mut().unwrap())?.extend(more);
                } else {
                    msg.insert("content_blocks".into(), blocks);
                    merged.push(msg);
                }
            }
            _ => merged.push(msg),
        }
    }
    Ok(merged)
}

fn blocks_mut(m: &mut Msg) -> Result<&mut Vec<Value>, String> {
    m.get_mut("content_blocks").and_then(|b| b.as_array_mut()).ok_or_else(|| "content_blocks must be a list".to_string())
}

/// sort_tool_results_by_call_order: a user turn's results in the order of the latest
/// assistant turn's calls (stable; an unknown id sorts as 0)
fn sort_tool_results(messages: &mut [Msg]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        match role(msg) {
            Some("assistant") if truthy(msg.get("tool_calls")) => {
                order.clear();
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for (i, tc) in calls.iter().enumerate() {
                        let id = if truthy(tc.get("id")) { tc.get("id") } else { tc.get("function").and_then(|f| f.get("id")) };
                        if truthy(id) {
                            order.push((py_str(id.unwrap()), i));
                        }
                    }
                }
            }
            Some("user") if truthy(msg.get("content_blocks")) => {
                let Some(blocks) = msg.get_mut("content_blocks").and_then(|b| b.as_array_mut()) else { continue };
                let is_result = |b: &Value| b.get("type").and_then(|t| t.as_str()) == Some("tool_result");
                let mut results: Vec<Value> = blocks.iter().filter(|b| is_result(b)).cloned().collect();
                if results.len() > 1 && !order.is_empty() {
                    let rank = |b: &Value| {
                        let id = b.get("tool_use_id").map(py_str).unwrap_or_default();
                        // a dict keyed by id: a repeated id keeps its last index
                        order.iter().rev().find(|(k, _)| *k == id).map(|&(_, i)| i).unwrap_or(0)
                    };
                    results.sort_by_key(rank);
                    let mut it = results.into_iter();
                    for b in blocks.iter_mut() {
                        if is_result(b) {
                            *b = it.next().unwrap();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn validate_no_image_tokens(m: &Msg) -> Result<(), String> {
    if m.get("content").and_then(|c| c.as_str()).is_some_and(|c| c.contains(IMAGE)) {
        return Err(format!("Message content contains image special token '{IMAGE}'. Images should be provided as image content blocks."));
    }
    if m.get("reasoning_content").and_then(|c| c.as_str()).is_some_and(|c| c.contains(IMAGE)) {
        return Err(format!("reasoning_content contains image special token '{IMAGE}'"));
    }
    Ok(())
}

fn is_image_block(b: &Value) -> bool {
    matches!(b.get("type").and_then(|t| t.as_str()), Some("image") | Some("image_url"))
}

/// _extract_image: {"type": "image", "url" | "source" | "data": ...}
fn extract_image(b: &Msg) -> Result<Value, String> {
    let mut rec = Msg::new();
    rec.insert("type".into(), "image".into());
    if b.get("type").and_then(|t| t.as_str()) == Some("image_url") {
        let url = match b.get("image_url") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Object(o)) => o.get("url").cloned().unwrap_or(Value::String(String::new())),
            _ => Value::String(String::new()),
        };
        rec.insert("url".into(), url);
    } else {
        for k in ["source", "url", "data"] {
            if let Some(v) = b.get(k) {
                rec.insert(k.into(), v.clone());
            }
        }
    }
    if !["source", "url", "data"].iter().any(|k| truthy(rec.get(*k))) {
        return Err("Image block does not contain a valid source".into());
    }
    Ok(Value::Object(rec))
}

/// _process_image_blocks: images become placeholder text blocks, their records collected in
/// one ordered walk (tool results' lists included)
fn process_image_blocks(blocks: &[Value], images: &mut Vec<Value>) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for b in blocks {
        let Some(o) = b.as_object() else {
            out.push(b.clone());
            continue;
        };
        if is_image_block(b) {
            out.push(json!({ "type": "text", "text": IMAGE }));
            images.push(extract_image(o)?);
        } else if o.get("type").and_then(|t| t.as_str()) == Some("tool_result") && o.get("content").is_some_and(|c| c.is_array()) {
            let mut o = o.clone();
            let inner = process_image_blocks(o["content"].as_array().unwrap(), images)?;
            o.insert("content".into(), Value::Array(inner));
            out.push(Value::Object(o));
        } else {
            if o.get("type").and_then(|t| t.as_str()) == Some("text") {
                let text = or_empty(o.get("text"));
                if text.contains(IMAGE) {
                    let head: String = text.chars().take(100).collect();
                    return Err(format!("Text block contains image placeholder '{IMAGE}': '{head}'. Images should be separate content blocks."));
                }
            }
            out.push(b.clone());
        }
    }
    Ok(out)
}

fn process_image_messages(messages: Vec<Msg>) -> Result<(Vec<Msg>, Vec<Value>), String> {
    let mut images = Vec::new();
    let mut out = Vec::new();
    for mut msg in messages {
        validate_no_image_tokens(&msg)?;
        if msg.get("content").is_some_and(|c| c.is_array()) && !msg.contains_key("content_blocks") {
            let c = msg.shift_remove("content").unwrap();
            msg.insert("content_blocks".into(), c);
        }
        if truthy(msg.get("content_blocks")) {
            let blocks = process_image_blocks(array(&msg["content_blocks"], "content_blocks")?, &mut images)?;
            if !msg.get("content").is_some_and(|c| c.is_string()) {
                let texts: Vec<String> = blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .map(|b| text_of(b.get("text")))
                    .collect::<Result<_, _>>()?;
                msg.insert("content".into(), Value::String(texts.join("\n\n")));
            }
            msg.insert("content_blocks".into(), Value::Array(blocks));
        }
        out.push(msg);
    }
    Ok((out, images))
}

// ---------------------------------------------------------------- rendering

/// find_last_user_index: a mid-conversation system message counts, a user turn of tool
/// results alone does not
fn last_user(messages: &[Msg]) -> i64 {
    for idx in (0..messages.len()).rev() {
        let m = &messages[idx];
        if role(m) == Some("user") || (role(m) == Some("system") && idx > 0) {
            if let Some(blocks) = m.get("content_blocks").filter(|b| truthy(Some(b))) {
                let shown = blocks.as_array().is_some_and(|a| {
                    a.iter().any(|b| matches!(b.get("type").and_then(|t| t.as_str()), Some("text") | Some("image") | Some("image_url")))
                });
                if !shown {
                    continue;
                }
            }
            return idx as i64;
        }
    }
    -1
}

fn render_message(index: usize, msgs: &[Msg], thinking: bool, drop_thinking: bool, effort: u32) -> Result<String, String> {
    let msg = &msgs[index];
    let last = last_user(msgs);
    let r = role(msg);
    let content = msg.get("content");
    let tools = match msg.get("tools") {
        t if truthy(t) => Some(tools_from_openai_format(array(t.unwrap(), "tools")?)?),
        _ => None,
    };
    let response_format = msg.get("response_format").filter(|v| truthy(Some(v)));
    let tool_calls = match msg.get("tool_calls") {
        t if truthy(t) => Some(tool_calls_from_openai_format(array(t.unwrap(), "tool_calls")?)?),
        _ => None,
    };

    let effort_prompt = if index == 0 && thinking {
        format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
    } else {
        String::new()
    };
    let mut p = String::new();
    if index == 0 && (!effort_prompt.is_empty() || r == Some("system")) {
        p.push_str(SYSTEM);
    }
    p.push_str(&effort_prompt);
    match r {
        Some("system") => {
            if index > 0 {
                p.push_str(SYSTEM);
            }
            p.push_str(&or_empty(content));
            if let Some(t) = &tools {
                p.push_str("\n\n");
                p.push_str(&render_tools(t));
            }
            if let Some(rf) = response_format {
                p.push_str("\n\n");
                p.push_str(RESPONSE_FORMAT);
                p.push_str(&py_json(rf));
            }
        }
        Some("user") => {
            p.push_str(USER);
            match msg.get("content_blocks").filter(|b| truthy(Some(b))) {
                Some(blocks) => {
                    let mut parts = Vec::new();
                    for b in array(blocks, "content_blocks")? {
                        object(b, "a content block")?;
                        match b.get("type").and_then(|t| t.as_str()) {
                            Some("text") => parts.push(text_of(b.get("text"))?),
                            Some("tool_result") => {
                                let c = b.get("content").cloned().unwrap_or(Value::String(String::new()));
                                let c = match &c {
                                    Value::Array(inner) => {
                                        let texts: Vec<String> = inner
                                            .iter()
                                            .map(|x| match x.get("type").and_then(|t| t.as_str()) {
                                                Some("text") => text_of(x.get("text")),
                                                _ => Ok(format!("[Unsupported {}]", block_type(x))),
                                            })
                                            .collect::<Result<_, _>>()?;
                                        texts.join("\n\n")
                                    }
                                    other => py_str(other),
                                };
                                parts.push(format!("<tool_result>{c}</tool_result>"));
                            }
                            _ => parts.push(format!("[Unsupported {}]", block_type(b))),
                        }
                    }
                    p.push_str(&parts.join("\n\n"));
                }
                None => p.push_str(&or_empty(content)),
            }
        }
        Some("latest_reminder") => {
            p.push_str(LATEST_REMINDER);
            p.push_str(&py_str(content.unwrap_or(&Value::Null)));
        }
        Some("tool") => {
            return Err("deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()".into());
        }
        Some("assistant") => {
            let mut tc = String::new();
            if let Some(calls) = &tool_calls {
                let invokes: Vec<String> = calls
                    .iter()
                    .map(|c| Ok(format!("<{DSML} invoke name=\"{}\">\n{}\n</{DSML} invoke>", tool_name_for_encoding(c)?, encode_arguments(c))))
                    .collect::<Result<_, String>>()?;
                tc = format!("\n\n<{DSML} calls>\n{}\n</{DSML} calls>", invokes.join("\n"));
            }
            let prev_has_task = index > 0 && get(&msgs[index - 1], "task").is_some();
            if thinking && !prev_has_task && (!drop_thinking || index as i64 > last) {
                p.push_str(&or_empty(msg.get("reasoning_content")));
                p.push_str(THINK_END);
            }
            p.push_str(&or_empty(content));
            p.push_str(&tc);
            if !truthy(msg.get("wo_eos")) {
                p.push_str(EOS);
            }
        }
        other => return Err(format!("Unknown role: {}", other.unwrap_or("None"))),
    }

    // the transition: a generation header (or a task's token) after the turn the model answers
    if index + 1 < msgs.len() && !matches!(role(&msgs[index + 1]), Some("assistant") | Some("latest_reminder")) {
        return Ok(p);
    }
    if let Some(task) = get(msg, "task") {
        let t = py_str(task);
        let Some(&(_, tok)) = TASKS.iter().find(|(k, _)| *k == t) else {
            let valid: Vec<&str> = TASKS.iter().map(|(k, _)| *k).collect();
            return Err(format!("Invalid task: '{t}'. Valid tasks are: {valid:?}"));
        };
        if t != "action" {
            p.push_str(tok);
        } else {
            p.push_str(ASSISTANT);
            p.push_str(if thinking { THINK_START } else { THINK_END });
            p.push_str(tok);
        }
    } else if r == Some("user") || (r == Some("system") && index > 0) {
        p.push_str(ASSISTANT);
        let open = thinking && (!drop_thinking || index as i64 >= last);
        p.push_str(if open { THINK_START } else { THINK_END });
    }
    Ok(p)
}

/// _drop_thinking_messages: before the last user turn, assistant turns lose their reasoning
/// and roles other than user / system / tool / latest_reminder / direct_search_results go
fn drop_thinking_messages(messages: Vec<Msg>) -> Vec<Msg> {
    let last = last_user(&messages);
    let keep = ["user", "system", "tool", "latest_reminder", "direct_search_results"];
    let mut out = Vec::new();
    for (idx, mut m) in messages.into_iter().enumerate() {
        let r = role(&m).map(String::from);
        if r.as_deref().is_some_and(|r| keep.contains(&r)) || idx as i64 >= last {
            out.push(m);
        } else if r.as_deref() == Some("assistant") {
            m.shift_remove("reasoning_content");
            out.push(m);
        }
    }
    out
}

/// A rendered prompt and the image records of its placeholders, in prompt order.
#[derive(Debug)]
pub struct Encoded {
    pub prompt: String,
    pub images: Vec<Value>,
}

/// encode_messages(messages, "thinking" if thinking else "chat", reasoning_effort=effort) with
/// the default BOS and drop_thinking (off once any message carries tools).
pub fn encode_messages(messages: &[Value], thinking: bool, effort: u32) -> Result<Encoded, String> {
    if !(1..=100).contains(&effort) {
        return Err(format!("Invalid reasoning effort for deepseek_v41: {effort}, should be int within [1,100] or ['low', 'high', 'max']"));
    }
    let msgs: Vec<Msg> = messages.iter().map(|m| object(m, "a message").cloned()).collect::<Result<_, _>>()?;
    for m in &msgs {
        validate_no_image_tokens(m)?;
    }
    let mut merged = merge_tool_messages(msgs)?;
    sort_tool_results(&mut merged);
    let (msgs, images) = process_image_messages(merged)?;
    let drop = !msgs.iter().any(|m| truthy(m.get("tools")));
    let msgs = if thinking && drop { drop_thinking_messages(msgs) } else { msgs };
    let mut prompt = String::from(BOS);
    for i in 0..msgs.len() {
        prompt.push_str(&render_message(i, &msgs, thinking, drop, effort)?);
    }
    Ok(Encoded { prompt, images })
}

// ---------------------------------------------------------------- the text-only interface

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// an assistant turn's reasoning (rendered only for turns after the last user message)
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message { role, content: content.into(), reasoning: None }
    }
}

/// encode_messages for text turns; `effort` None is "high" (75).
pub fn encode(messages: &[Message], thinking: bool, effort: Option<u32>) -> Result<String, String> {
    let msgs: Vec<Value> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let mut v = json!({ "role": role, "content": m.content });
            if let Some(r) = &m.reasoning {
                v["reasoning_content"] = Value::String(r.clone());
            }
            v
        })
        .collect();
    Ok(encode_messages(&msgs, thinking, effort.unwrap_or(75))?.prompt)
}

// ---------------------------------------------------------------- parsing an answer

/// A tool call as the model wrote it; `arguments` is the JSON object text (string="false"
/// values go in as written, unchecked, as the reference parser does).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub namespace: Option<String>,
    pub arguments: String,
}

impl ToolCall {
    /// the name as the client declared it: namespace::name
    pub fn full_name(&self) -> String {
        match &self.namespace {
            Some(n) => format!("{n}::{}", self.name),
            None => self.name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parsed {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

/// _read_until_stop: (index past the earliest stop, the text before it, the stop)
fn read_until<'a>(index: usize, text: &'a str, stops: &[&'static str]) -> (usize, &'a str, Option<&'static str>) {
    let mut best: Option<(usize, &'static str)> = None;
    for &s in stops {
        if let Some(p) = text[index..].find(s) {
            if best.is_none_or(|(b, _)| index + p < b) {
                best = Some((index + p, s));
            }
        }
    }
    match best {
        Some((p, s)) => (p + s.len(), &text[index..p], Some(s)),
        None => (text.len(), &text[index..], None),
    }
}

/// `$` without MULTILINE also matches before one final "\n": the candidates for the end of a
/// group followed by `lit` and then `$`, shortest group first (a lazy group's order)
fn ends_before(rest: &str, lit: &str) -> Option<usize> {
    let nl = format!("{lit}\n");
    if rest.ends_with(&nl) {
        Some(rest.len() - nl.len())
    } else if rest.ends_with(lit) {
        Some(rest.len() - lit.len())
    } else {
        None
    }
}

/// `^\s*name="(.*?)">\n$`
fn match_tool_name(s: &str) -> Option<&str> {
    let rest = s.trim_start_matches(char::is_whitespace).strip_prefix("name=\"")?;
    ends_before(rest, "\">\n").map(|e| &rest[..e])
}

/// `^ name="(.*?)" string="(true|false)">(.*?)<$`
fn match_param(s: &str) -> Option<(&str, &'static str, &str)> {
    let rest = s.strip_prefix(" name=\"")?;
    let mut from = 0;
    while let Some(off) = rest[from..].find("\" string=\"") {
        let at = from + off;
        let tail = &rest[at + "\" string=\"".len()..];
        for flag in ["true", "false"] {
            if let Some(v) = tail.strip_prefix(flag).and_then(|t| t.strip_prefix("\">")) {
                if let Some(e) = ends_before(v, "<") {
                    return Some((&rest[..at], flag, &v[..e]));
                }
            }
        }
        from = at + 1;
    }
    None
}

fn parse_tool_calls(mut index: usize, text: &str) -> Result<(usize, Vec<ToolCall>), String> {
    let mut calls = Vec::new();
    while index < text.len() {
        let (i, skipped, stop) = read_until(index, text, &[INVOKE_START, CALLS_END]);
        index = i;
        if skipped != ">\n" {
            return Err(format!("Tool call format error: expected '>\\n' but got '{skipped}'"));
        }
        if stop == Some(CALLS_END) {
            break;
        }
        if stop.is_none() {
            return Err("Missing special token in tool calls".into());
        }
        let (i, name_text, mut stop) = read_until(index, text, &[PARAM_START, INVOKE_END]);
        index = i;
        let name = match_tool_name(name_text).ok_or_else(|| format!("Tool name format error: '{name_text}'"))?;
        let mut args: Vec<(String, String, &str)> = Vec::new();
        while stop == Some(PARAM_START) {
            let (i, param, _) = read_until(index, text, &[PARAM_END]);
            index = i;
            let (k, flag, v) = match_param(param).ok_or_else(|| format!("Parameter format error: '{param}'"))?;
            if args.iter().any(|(a, _, _)| a == k) {
                return Err(format!("Duplicate parameter name: '{k}'"));
            }
            args.push((k.to_string(), v.to_string(), flag));
            let (i, between, s) = read_until(index, text, &[PARAM_START, INVOKE_END]);
            (index, stop) = (i, s);
            if between != ">\n" {
                return Err(format!("Parameter format error: expected '>\\n' but got '{between}'"));
            }
        }
        // decode_dsml_to_arguments
        let body: Vec<String> = args
            .iter()
            .map(|(k, v, flag)| {
                let v = if *flag == "true" { py_json(&Value::String(v.clone())) } else { v.clone() };
                format!("{}: {v}", py_json(&Value::String(k.clone())))
            })
            .collect();
        let (namespace, name) = split_tool_name(name, None)?;
        calls.push(ToolCall { name, namespace, arguments: format!("{{{}}}", body.join(", ")) });
    }
    Ok((index, calls))
}

/// parse_message_from_completion_text: an answer (ending in EOS) as reasoning, content and
/// tool calls, or why it does not have the reference's shape.
pub fn parse_completion(text: &str, thinking: bool) -> Result<Parsed, String> {
    let (mut index, mut reasoning) = (0, "");
    if thinking {
        let (i, r, stop) = read_until(index, text, &[THINK_END, TOOL_CALLS_START]);
        if stop != Some(THINK_END) {
            return Err("Invalid thinking format: missing </think>".into());
        }
        (index, reasoning) = (i, r);
    }
    let (i, content, mut stop) = read_until(index, text, &[EOS, TOOL_CALLS_START]);
    index = i;
    let mut tool_calls = Vec::new();
    if stop == Some(TOOL_CALLS_START) {
        let (i, calls) = parse_tool_calls(index, text)?;
        tool_calls = calls;
        let (i, after, s2) = read_until(i, text, &[EOS]);
        if !after.is_empty() {
            return Err("Unexpected content after tool calls".into());
        }
        (index, stop) = (i, s2);
    } else if stop != Some(EOS) {
        return Err("Invalid format: missing EOS token".into());
    }
    if index != text.len() || !matches!(stop, Some(EOS) | None) {
        return Err("Unexpected content at end".into());
    }
    for sp in [BOS, EOS, THINK_START, THINK_END, DSML] {
        if content.contains(sp) || reasoning.contains(sp) {
            return Err(format!("Unexpected special token '{sp}' in content"));
        }
    }
    Ok(Parsed { content: content.into(), reasoning: reasoning.into(), tool_calls })
}

/// server/app.py's _parse_tool_calls_tolerant, for a block the strict parser refused (a value
/// written into the `string` attribute, prose after the block, an answer cut off): each
/// `invoke`'s name and its arguments as JSON - `string="true"` values as text up to the first
/// `<`, `string="false"` ones decoded where they are JSON, an attribute-only value as text.
/// The whole text is searched, reasoning included. Empty when nothing parses.
pub fn parse_tool_calls_tolerant(text: &str) -> Vec<(String, String)> {
    const INV: &str = "<｜DSML｜ invoke name=\"";
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(off) = text[pos..].find(INV) {
        let ns = pos + off + INV.len();
        let Some(q) = text[ns..].find('"') else { break };
        let name = &text[ns..ns + q];
        // \s*>?\n?
        let mut i = ns + q + 1;
        i = text.len() - text[i..].trim_start_matches(char::is_whitespace).len();
        if text[i..].starts_with('>') {
            i += 1;
        }
        if text[i..].starts_with('\n') {
            i += 1;
        }
        // the body runs to the next invoke, the end of the block or the end
        let end = ["<｜DSML｜ invoke ", "</｜DSML｜ calls>"]
            .iter()
            .filter_map(|m| text[i..].find(m).map(|p| i + p))
            .min()
            .unwrap_or(text.len());
        if !name.is_empty() {
            out.push((name.to_string(), py_json(&Value::Object(tolerant_args(&text[i..end])))));
        }
        pos = end;
    }
    out
}

fn tolerant_args(body: &str) -> Msg {
    const PAR: &str = "<｜DSML｜ parameter name=\"";
    // (key, flag, value) of `name="k" string="true|false"\s*>v<`; (key, value) of the
    // attribute form `name="k" string="v"\s*>`
    let (mut spec, mut attr) = (Vec::new(), Vec::new());
    let mut pos = 0;
    while let Some(off) = body[pos..].find(PAR) {
        let ks = pos + off + PAR.len();
        let Some(q) = body[ks..].find('"') else { break };
        let (k, after) = (&body[ks..ks + q], &body[ks + q..]);
        pos = ks;
        let Some(r) = after.strip_prefix("\" string=\"") else { continue };
        let flag = if r.starts_with("true") { Some("true") } else if r.starts_with("false") { Some("false") } else { None };
        let Some(flag) = flag else { continue };
        let Some(r) = r[flag.len()..].strip_prefix('"') else { continue };
        let Some(r) = r.trim_start_matches(char::is_whitespace).strip_prefix('>') else { continue };
        let vs = body.len() - r.len();
        let Some(lt) = body[vs..].find('<') else { break };
        spec.push((k, flag, &body[vs..vs + lt]));
        pos = vs + lt + 1;
    }
    pos = 0;
    while let Some(off) = body[pos..].find(PAR) {
        let ks = pos + off + PAR.len();
        let Some(q) = body[ks..].find('"') else { break };
        let k = &body[ks..ks + q];
        pos = ks;
        let Some(r) = body[ks + q..].strip_prefix("\" string=\"") else { continue };
        let vs = body.len() - r.len();
        // the shortest value followed by `"\s*>`
        let mut from = vs;
        let mut found = None;
        while let Some(qo) = body[from..].find('"') {
            let qa = from + qo;
            if let Some(rest) = body[qa + 1..].trim_start_matches(char::is_whitespace).strip_prefix('>') {
                found = Some((qa, body.len() - rest.len()));
                break;
            }
            from = qa + 1;
        }
        let Some((qa, end)) = found else { continue };
        attr.push((k, &body[vs..qa]));
        pos = end;
    }
    let mut args = Msg::new();
    for &(k, flag, v) in &spec {
        let val = if flag == "true" {
            Value::String(v.into())
        } else {
            serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.into()))
        };
        args.insert(k.into(), val);
    }
    for (k, v) in attr {
        if !k.is_empty() && !spec.iter().any(|(s, _, _)| *s == k) && !args.contains_key(k) {
            args.insert(k.into(), Value::String(v.into()));
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_mode_single_user_turn_is_the_bench_prompt() {
        let p = encode(&[Message::new(Role::User, "[req 1] hi")], false, None).unwrap();
        assert_eq!(p, "<｜begin▁of▁sentence｜><｜User｜>[req 1] hi<｜Assistant｜></think>");
    }

    #[test]
    fn thinking_mode_renders_the_effort_under_a_system_token() {
        let p = encode(&[Message::new(Role::User, "q")], true, Some(60)).unwrap();
        assert_eq!(p, "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 60 (range 1-100, the higher the value, the more \
                       thorough the reasoning)\n\n<｜User｜>q<｜Assistant｜><think>");
    }

    #[test]
    fn floats_are_written_as_python_repr() {
        for (f, want) in [(1e10, "10000000000.0"), (1.5, "1.5"), (1e16, "1e+16"), (1e-5, "1e-05"), (0.0001, "0.0001"),
                          (-2.5e-7, "-2.5e-07"), (123456789.125, "123456789.125"), (1.2345678901234568e17, "1.2345678901234568e+17"),
                          (0.1, "0.1"), (100.0, "100.0")] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }
}
