//! Inkling 2's chat prompt and the router of its answers: the checkpoint's chat_template.jinja
//! by hand, and a parser for the messages the model writes back.
//!
//! render is what transformers' apply_chat_template makes of the template: jinja2's sandboxed
//! environment, trim_blocks, lstrip_blocks, and a `tojson` that is json.dumps with
//! ensure_ascii=False. That filter is the only one the template can run under: it calls
//! `tojson(sort_keys=true, separators=(",", ":"))`, and jinja2's own tojson takes neither
//! (jinja2 3.1.2: "do_tojson() got an unexpected keyword argument 'sort_keys'"), so jinja2's
//! HTML escapes (`\u003c` for `<`) never reach a prompt. The prompt, in order:
//!
//! - the tools, when there are some: `<|message_system|>tool_declare<|content_xml|>`, a JSON
//!   list of `{"description","name","parameters","type"}` (keys sorted at every depth, no
//!   spaces), `<|end_message|>`;
//! - the messages, each `ROLE[name]<|content_text|>text<|end_message|>` (a list content: one
//!   such message per part; an image part is `<|content_image|><|unused_200054|>`). A tool
//!   message's header carries its tool's name: its own `name`, else the name of every
//!   assistant tool call in the conversation whose id is its `tool_call_id` (every one: a
//!   client that reuses "call_0" in two turns gets both names, as the template writes them);
//! - an assistant turn: `<|message_model|><|content_thinking|>reasoning<|end_message|>`, its
//!   text, then each tool call as
//!   `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`,
//!   and `<|content_model_end_sampling|>`;
//! - once, before the first message that is not a system message (after the last when all
//!   are): `<|message_system|><|content_text|>Thinking effort level: N<|end_message|>`, N from
//!   the request's effort (EFFORTS; 0.9 when there is none; a number in [0, 0.99] as it is).
//!
//! There is no generation prompt and no begin-of-text token: the model writes its own
//! `<|message_model|>`. The template's value rules are kept where they decide bytes - jinja2's
//! Undefined for a missing key, Python truthiness, `str()` of a non-string, Python's `==` on
//! ids - and where it raises, render refuses with the template's message. It takes a tool
//! call's `arguments` as an object and refuses a string ("canonicalize upstream"): prepare is
//! that step for an OpenAI client's JSON strings.
//!
//! InklingParser routes the answer's token ids (every marker is a single token): a message is
//! `<|message_model|>`, a header (a tool call's name, else empty), a content token that says
//! what the body is, the body, `<|end_message|>`. `<|content_thinking|>` bodies are reasoning,
//! `<|content_invoke_tool_json|>` bodies are held and parsed when the answer is done, and
//! `<|content_text|>` and every other `<|content_*|>` kind are content. A stop id ends the
//! answer. parse_answer is the same router over an answer's text.

use std::borrow::Cow;
use std::ops::Deref;
use std::path::Path;

use serde_json::{json, Value};

use crate::chat::{py_float, py_json, py_loads, py_number};
pub use crate::glm::Channel;
use crate::glm::{py_strip, truthy};
use crate::{Error, Text};

pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
pub const USER: &str = "<|message_user|>";
pub const MODEL: &str = "<|message_model|>";
pub const SYSTEM: &str = "<|message_system|>";
pub const TOOL: &str = "<|message_tool|>";
pub const END_MESSAGE: &str = "<|end_message|>";
pub const TEXT: &str = "<|content_text|>";
pub const THINKING: &str = "<|content_thinking|>";
pub const IMAGE: &str = "<|content_image|>";
pub const XML: &str = "<|content_xml|>";
pub const TOOL_ERROR: &str = "<|content_tool_error|>";
pub const INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
/// the end of every assistant turn, and the model's end-of-sequence token
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";
/// the content parts the template renders as an image
pub const IMAGES: [&str; 3] = ["image", "input_image", "image_url"];
/// an image part's text in the prompt (processor_config.json's image_token): the token a
/// vision server expands into the image's tokens
pub const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";
const AUDIO: [&str; 3] = ["audio", "input_audio", "audio_url"];
/// The template's effort_map: a named effort's "Thinking effort level"
pub const EFFORTS: [(&str, f64); 7] =
    [("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7), ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)];
/// the level of a request without an effort ("high")
pub const DEFAULT_EFFORT: f64 = 0.9;

// ---------------------------------------------------------------- Python's values
//
// str() and == as glm.rs has them (private there; these are copies, to leave that file alone).

fn py_unprintable(c: char) -> bool {
    matches!(c, '\0'..='\u{1f}' | '\u{7f}'..='\u{a0}' | '\u{ad}' | '\u{1680}' | '\u{180e}' | '\u{2000}'..='\u{200f}'
                | '\u{2028}'..='\u{202f}' | '\u{205f}'..='\u{206f}' | '\u{3000}' | '\u{e000}'..='\u{f8ff}' | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}' | '\u{f0000}'..)
}

fn py_repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if py_unprintable(c) => {
                let n = c as u32;
                out.push_str(&if n < 0x100 { format!("\\x{n:02x}") } else if n < 0x10000 { format!("\\u{n:04x}") } else { format!("\\U{n:08x}") });
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => py_number(n),
        Value::String(s) => py_repr_str(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{}: {}", py_repr_str(k), py_repr(x))).collect::<Vec<_>>().join(", ")),
    }
}

/// What `~ v` and `{{ v }}` write: `str(v)`
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python's `==` on JSON values (True == 1, 1 == 1.0)
fn py_eq(a: &Value, b: &Value) -> bool {
    let num = |v: &Value| match v {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if (x.is_i64() || x.is_u64()) && (y.is_i64() || y.is_u64()) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, p)| y.get(k).is_some_and(|q| py_eq(p, q))),
        _ => matches!((num(a), num(b)), (Some(x), Some(y)) if x == y),
    }
}

/// `json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`: the template's
/// `tojson(sort_keys=true, separators=(",", ":"))`
pub fn sorted_json(v: &Value) -> String {
    let mut out = String::new();
    write_sorted(v, &mut out);
    out
}

fn write_sorted(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_number(n)),
        // json.dumps' escapes without ensure_ascii are serde_json's (chat.rs's write_py)
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap()),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_sorted(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            // Python orders str keys by code point, which is UTF-8's byte order
            let mut items: Vec<(&String, &Value)> = m.iter().collect();
            items.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (k, x)) in items.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push(':');
                write_sorted(x, out);
            }
            out.push('}');
        }
    }
}

// ---------------------------------------------------------------- the template

/// `v.key` as jinja2 reads it: a mapping's item. None is jinja2's Undefined: a missing key, or
/// any `v` that is not a mapping.
fn get<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.as_object().and_then(|m| m.get(k))
}

/// `v.key is defined and v.key`
fn get_truthy<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    get(v, k).filter(|x| truthy(x))
}

fn undefined(what: &str) -> String {
    format!("UndefinedError: {what}")
}

/// A named effort's level (EFFORTS)
pub fn effort_level(name: &str) -> Option<f64> {
    EFFORTS.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}

/// The macro `emit_thinking_effort`. `effort` is the template's `reasoning_effort`: a string
/// is a name (stripped; an unknown one raises, "0.5" too), anything else is `| float` of it
/// (a number itself, true 1.0, and 0.0 for what float() refuses), and none is 0.9. The level
/// prints as Python prints a float, and 0.0 as `0`.
fn effort_line(effort: Option<&Value>) -> Result<String, String> {
    let num = match effort {
        None | Some(Value::Null) => DEFAULT_EFFORT,
        Some(Value::String(s)) => effort_level(py_strip(s)).ok_or_else(|| format!("Unknown reasoning_effort: {s}"))?,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::Bool(b)) => *b as u8 as f64,
        Some(_) => 0.0,
    };
    if !(0.0..=0.99).contains(&num) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    let level = if num == 0.0 { "0".to_string() } else { py_float(num) };
    Ok(format!("{SYSTEM}{TEXT}Thinking effort level: {level}{END_MESSAGE}"))
}

/// The tools block: each tool's `function` (the tool itself without one) as a spec whose
/// missing or falsy description, parameters and type are "", {} and "function"
fn tool_declare(tools: &[Value]) -> Result<String, String> {
    let mut out = format!("{SYSTEM}tool_declare{XML}[");
    for (i, tool) in tools.iter().enumerate() {
        let f = get(tool, "function").unwrap_or(tool);
        // json.dumps refuses jinja2's Undefined
        let name = get(f, "name").ok_or_else(|| format!("TypeError: tools[{i}] has no name"))?;
        let or = |v: Option<&Value>, default: Value| sorted_json(v.unwrap_or(&default));
        if i > 0 {
            out.push(',');
        }
        // the spec's keys, sorted
        out.push_str(&format!("{{\"description\":{},\"name\":{},\"parameters\":{},\"type\":{}}}",
                              or(get_truthy(f, "description"), json!("")), sorted_json(name),
                              or(get_truthy(f, "parameters"), json!({})), or(get_truthy(tool, "type"), json!("function"))));
    }
    out.push(']');
    out.push_str(END_MESSAGE);
    Ok(out)
}

fn role_token(role: &str) -> Option<&'static str> {
    match role {
        "user" => Some(USER),
        "assistant" => Some(MODEL),
        "system" => Some(SYSTEM),
        "tool" => Some(TOOL),
        _ => None,
    }
}

/// (id, function) of every assistant tool call that has an id, in conversation order: what a
/// tool message's `tool_call_id` is matched against
type Calls<'a> = Vec<(&'a Value, Option<&'a Value>)>;

fn calls_with_ids(messages: &[Value]) -> Result<Calls<'_>, String> {
    let mut out = Vec::new();
    for prev in messages {
        if get(prev, "role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        match get_truthy(prev, "tool_calls") {
            Some(Value::Array(calls)) => out.extend(calls.iter().filter_map(|tc| get(tc, "id").map(|id| (id, get(tc, "function"))))),
            // a mapping iterates as its keys and a string as its characters: nothing with an id
            None | Some(Value::Object(_) | Value::String(_)) => {}
            Some(other) => return Err(format!("TypeError: tool_calls is not iterable: {}", py_repr(other))),
        }
    }
    Ok(out)
}

/// The prompt text for OpenAI-shaped `messages` and `tools` at a named effort (EFFORTS; none:
/// 0.9), with glm::render's signature. A tool call's `arguments` must already be an object
/// (prepare). Image parts render as the template renders them, IMAGE_PLACEHOLDER each; nothing
/// serves image input yet, and image_parts counts them for a server to refuse.
pub fn render(messages: &[Value], tools: Option<&Value>, effort: Option<&str>) -> Result<String, String> {
    let effort = effort.map(|e| Value::String(e.to_string()));
    render_with_effort(messages, tools, effort.as_ref())
}

/// render with the template's whole `reasoning_effort`: a name, or a number in [0, 0.99]
pub fn render_with_effort(messages: &[Value], tools: Option<&Value>, effort: Option<&Value>) -> Result<String, String> {
    let mut out = String::new();
    match tools {
        Some(Value::Array(tools)) if !tools.is_empty() => out.push_str(&tool_declare(tools)?),
        Some(other) if truthy(other) => return Err("TypeError: tools must be a list".into()),
        _ => {}
    }
    let mut effort_emitted = false;
    let mut calls: Option<Calls> = None;
    for m in messages {
        let role = match get(m, "role") {
            Some(Value::String(r)) if role_token(r).is_some() => r.as_str(),
            Some(r @ (Value::Array(_) | Value::Object(_))) => return Err(format!("TypeError: unhashable message role: {}", py_repr(r))),
            other => return Err(format!("Unknown message role: {}", other.map(py_str).unwrap_or_default())),
        };
        if !effort_emitted && role != "system" {
            out.push_str(&effort_line(effort)?);
            effort_emitted = true;
        }
        let assistant = role == "assistant";

        let mut header = String::from(role_token(role).unwrap_or_default());
        if role == "tool" {
            if let Some(name) = get_truthy(m, "name") {
                header.push_str(&py_str(name));
            } else if let Some(id) = get_truthy(m, "tool_call_id") {
                if calls.is_none() {
                    calls = Some(calls_with_ids(messages)?);
                }
                for (call_id, function) in calls.iter().flatten() {
                    if py_eq(call_id, id) {
                        let f = function.ok_or_else(|| undefined("a tool call has no attribute 'function'"))?;
                        if let Some(name) = get(f, "name") {
                            header.push_str(&py_str(name));
                        }
                    }
                }
            }
        } else if assistant {
            if let Some(name) = get_truthy(m, "name") {
                header.push_str(&py_str(name));
            }
        }

        let reasoning = get_truthy(m, "reasoning_content").filter(|_| assistant);
        if let Some(r) = reasoning {
            out.push_str(&format!("{header}{THINKING}{}{END_MESSAGE}", py_str(r)));
        }
        let tool_calls = get_truthy(m, "tool_calls").filter(|_| assistant);

        let text = |out: &mut String, s: &str| out.push_str(&format!("{header}{TEXT}{s}{END_MESSAGE}"));
        match get(m, "content") {
            // an assistant's empty text beside reasoning or tool calls is no message
            Some(Value::String(s)) => {
                if !s.is_empty() || (reasoning.is_none() && tool_calls.is_none()) {
                    text(&mut out, s);
                }
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    let ty = get(part, "type");
                    let ty_in = |set: &[&str]| ty.and_then(|t| t.as_str()).is_some_and(|t| set.contains(&t));
                    if let Value::String(s) = part {
                        text(&mut out, s);
                    } else if ty.is_none() || ty_in(&["text", "input_text"]) {
                        // a part that is no mapping has no type either: an empty text
                        text(&mut out, get(part, "text").and_then(|t| t.as_str()).unwrap_or(""));
                    } else if ty_in(&IMAGES) {
                        if role != "user" && role != "tool" {
                            return Err("Image content is only supported for user and tool messages".into());
                        }
                        out.push_str(&format!("{header}{IMAGE}{IMAGE_PLACEHOLDER}{END_MESSAGE}"));
                    } else if ty_in(&AUDIO) {
                        return Err("Inkling 2 does not accept audio input".into());
                    } else {
                        return Err(format!("Unsupported content part type: {}", ty.map(py_str).unwrap_or_default()));
                    }
                }
            }
            // a mapping iterates as its keys, each a string part
            Some(Value::Object(parts)) => parts.keys().for_each(|k| text(&mut out, k)),
            Some(other) if truthy(other) => return Err(format!("TypeError: content is not iterable: {}", py_repr(other))),
            _ => {}
        }

        if let Some(tool_calls) = tool_calls {
            let Value::Array(tool_calls) = tool_calls else {
                return Err("TypeError: tool_calls must be a list".into());
            };
            for tc in tool_calls {
                let f = get(tc, "function").ok_or_else(|| undefined("a tool call has no attribute 'function'"))?;
                let Some(Value::String(name)) = get(f, "name") else {
                    return Err("tool call function name must be a string".into());
                };
                if let Some(own) = get_truthy(m, "name").filter(|own| own.as_str() != Some(name.as_str())) {
                    return Err(format!("assistant name {} does not match its tool call {name}", py_str(own)));
                }
                let args = match get_truthy(f, "arguments") {
                    None => "{}".to_string(),
                    Some(Value::String(_)) => return Err("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream".into()),
                    Some(args @ Value::Object(_)) => sorted_json(args),
                    Some(_) => return Err("tool call arguments must be an object".into()),
                };
                out.push_str(&format!("{MODEL}{name}{INVOKE_TOOL_JSON}{{\"name\":{},\"args\":{args}}}{END_MESSAGE}",
                                      serde_json::to_string(name).unwrap()));
            }
        }
        if assistant {
            out.push_str(END_SAMPLING);
        }
    }
    if !effort_emitted {
        out.push_str(&effort_line(effort)?);
    }
    Ok(out)
}

/// The step the template leaves upstream: an OpenAI client sends a tool call's `arguments` as
/// a JSON string, and the template takes an object. Each such string is parsed, glm.rs's rule
/// (glm_app's): text that is not JSON becomes `{"input": raw}`, so an answer whose arguments
/// were cut off does not make its conversation unrenderable. Everything else is left for the
/// template to judge: a string that parses to a list is refused there, an empty one is `{}`
/// there. Borrows when no message has such a string.
pub fn prepare(messages: &[Value]) -> Cow<'_, [Value]> {
    let string_args = |tc: &Value| get(tc, "function").and_then(|f| get(f, "arguments")).is_some_and(|a| a.is_string());
    let needs = |m: &Value| get(m, "tool_calls").and_then(|c| c.as_array()).is_some_and(|c| c.iter().any(string_args));
    if !messages.iter().any(needs) {
        return Cow::Borrowed(messages);
    }
    let mut out = messages.to_vec();
    for m in out.iter_mut().filter(|m| needs(m)) {
        for tc in m["tool_calls"].as_array_mut().into_iter().flatten() {
            if let Some(args) = tc.get_mut("function").and_then(|f| f.get_mut("arguments")) {
                let parsed = match &*args {
                    Value::String(s) if !s.is_empty() => Some(py_loads(s).unwrap_or_else(|_| json!({ "input": s }))),
                    _ => None,
                };
                if let Some(parsed) = parsed {
                    *args = parsed;
                }
            }
        }
    }
    Cow::Owned(out)
}

/// How many image parts render writes a placeholder for: the parts of list contents whose
/// type is one of IMAGES
pub fn image_parts(messages: &[Value]) -> usize {
    messages.iter().filter_map(|m| get(m, "content").and_then(|c| c.as_array())).flatten()
        .filter(|p| get(p, "type").and_then(|t| t.as_str()).is_some_and(|t| IMAGES.contains(&t)))
        .count()
}

// ---------------------------------------------------------------- the answer

/// The marker ids the parser routes on and the stop ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InklingIds {
    /// `<|message_model|>`: a message's header begins
    pub model: u32,
    /// `<|end_message|>`
    pub end: u32,
    /// `<|content_thinking|>`: the body is reasoning
    pub thinking: u32,
    /// `<|content_text|>`: the body is content
    pub text: u32,
    /// `<|content_invoke_tool_json|>`: the body is a tool call
    pub tool_json: u32,
    /// every other `<|content_*|>` token the tokenizer has (image, tool_error, xml,
    /// invoke_tool_text): the body is content
    pub other: Vec<u32>,
    /// IMAGE_PLACEHOLDER
    pub image: u32,
    /// `<|content_model_end_sampling|>` first, then the eos_token_id of generation_config.json
    /// and config.json
    pub stop: Vec<u32>,
}

/// `<|content_NAME|>`, NAME in lowercase letters, digits and `_`
fn content_marker_len(s: &str) -> Option<usize> {
    let name = s.strip_prefix("<|content_")?;
    let end = name.find("|>")?;
    let ok = end > 0 && name[..end].bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    ok.then_some("<|content_".len() + end + 2)
}

impl InklingIds {
    pub fn load(model_dir: &Path, text: &Text) -> Result<Self, Error> {
        let id = |t: &str| text.tokenizer.token_to_id(t).ok_or_else(|| format!("the tokenizer has no {t} token"));
        let mut stop = vec![id(END_SAMPLING)?];
        for file in ["generation_config.json", "config.json"] {
            let path = model_dir.join(file);
            if !path.exists() {
                continue;
            }
            let cfg: Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let eos: Vec<u64> = match &cfg["eos_token_id"] {
                Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).collect(),
                Value::Number(n) => n.as_u64().into_iter().collect(),
                _ => Vec::new(),
            };
            for e in eos {
                if !stop.contains(&(e as u32)) {
                    stop.push(e as u32);
                }
            }
        }
        let (thinking, text_id, tool_json) = (id(THINKING)?, id(TEXT)?, id(INVOKE_TOOL_JSON)?);
        let mut other: Vec<u32> = text.tokenizer.get_added_tokens_decoder().iter()
            .filter(|(i, t)| content_marker_len(&t.content) == Some(t.content.len()) && ![thinking, text_id, tool_json].contains(i) && !stop.contains(i))
            .map(|(i, _)| *i)
            .collect();
        other.sort_unstable();
        Ok(InklingIds { model: id(MODEL)?, end: id(END_MESSAGE)?, thinking, text: text_id, tool_json, other, image: id(IMAGE_PLACEHOLDER)?, stop })
    }

    fn mark(&self, tok: u32) -> Option<Mark> {
        Some(if self.stop.contains(&tok) {
            Mark::Stop
        } else if tok == self.model {
            Mark::Model
        } else if tok == self.end {
            Mark::End
        } else if tok == self.thinking {
            Mark::Kind(Kind::Thinking)
        } else if tok == self.text {
            Mark::Kind(Kind::Text)
        } else if tok == self.tool_json {
            Mark::Kind(Kind::ToolJson)
        } else if self.other.contains(&tok) {
            Mark::Kind(Kind::Other)
        } else {
            return None;
        })
    }
}

/// What a body is, by the content token that opened it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    ToolJson,
    /// a content kind with no rule of its own: content
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    Model,
    Kind(Kind),
    End,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// between messages
    Idle,
    /// after `<|message_model|>`, before the content token
    Header,
    Body(Kind),
}

/// One tool-call message as the model wrote it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawCall {
    /// the text between `<|message_model|>` and `<|content_invoke_tool_json|>`: the tool's name
    pub header: String,
    /// the body: `{"name":NAME,"args":{...}}`
    pub json: String,
    /// `<|end_message|>` ended it (false: the answer was cut inside it, or another marker
    /// ended it)
    pub closed: bool,
}

/// An answer routed so far: InklingParser's state without the token ids.
///
/// The model is trusted to write the format and not relied on to: a marker ends whatever was
/// open (a body without its `<|end_message|>` ends at the next `<|message_model|>`, content
/// token or stop), and no text is dropped except a tool call's header. Text between messages
/// is content. A header that no content token follows - `<|end_message|>`, another
/// `<|message_model|>`, a stop, or the end of the answer came first - is content too: a model
/// that never writes a content token still shows what it wrote.
#[derive(Clone, Debug)]
pub struct Routed {
    state: State,
    header: String,
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<RawCall>,
    /// a stop marker ended the answer; nothing after it is routed
    pub ended: bool,
}

impl Routed {
    fn new() -> Self {
        Routed { state: State::Idle, header: String::new(), reasoning: String::new(), content: String::new(), calls: Vec::new(), ended: false }
    }

    fn emit(&mut self, chan: Channel, s: &str, out: &mut Vec<(Channel, String)>) {
        if s.is_empty() {
            return;
        }
        match chan {
            Channel::Reasoning => self.reasoning.push_str(s),
            Channel::Content => self.content.push_str(s),
        }
        out.push((chan, s.to_string()));
    }

    fn text(&mut self, s: &str, out: &mut Vec<(Channel, String)>) {
        if self.ended {
            return;
        }
        match self.state {
            State::Header => self.header.push_str(s),
            State::Body(Kind::ToolJson) => {
                if let Some(c) = self.calls.last_mut() {
                    c.json.push_str(s);
                }
            }
            State::Body(Kind::Thinking) => self.emit(Channel::Reasoning, s, out),
            State::Idle | State::Body(Kind::Text | Kind::Other) => self.emit(Channel::Content, s, out),
        }
    }

    /// The header was no header: no content token followed it
    fn release_header(&mut self, out: &mut Vec<(Channel, String)>) {
        let header = std::mem::take(&mut self.header);
        if self.state == State::Header {
            self.emit(Channel::Content, &header, out);
        }
    }

    fn mark(&mut self, m: Mark, out: &mut Vec<(Channel, String)>) {
        if self.ended {
            return;
        }
        if let (State::Body(Kind::ToolJson), Mark::End, Some(c)) = (self.state, m, self.calls.last_mut()) {
            c.closed = true;
        }
        self.state = match m {
            Mark::Kind(kind) => {
                let header = std::mem::take(&mut self.header);
                if kind == Kind::ToolJson {
                    self.calls.push(RawCall { header, json: String::new(), closed: false });
                }
                State::Body(kind)
            }
            other => {
                self.release_header(out);
                self.ended = other == Mark::Stop;
                if other == Mark::Model { State::Header } else { State::Idle }
            }
        };
    }

    /// The answer is over (a stop, or it was cut): a header still open is content
    fn close(&mut self, out: &mut Vec<(Channel, String)>) {
        self.release_header(out);
        if self.state == State::Header {
            self.state = State::Idle;
        }
    }

    /// Each tool call's body
    pub fn call_texts(&self) -> Vec<String> {
        self.calls.iter().map(|c| c.json.clone()).collect()
    }

    /// (name, arguments JSON) of each tool call, one cut off mid-body too (parse_tool_call)
    pub fn tool_calls(&self) -> Vec<(String, String)> {
        self.calls.iter().map(|c| parse_tool_call(&c.header, &c.json)).collect()
    }
}

/// One tool call, `{"name":NAME,"args":{...}}` under its header: the JSON's name when it is a
/// string, else the header (stripped), and its args as json.dumps(ensure_ascii=False) writes
/// them (glm.rs's and the DeepSeek router's bytes for `arguments`; `{}` without args). A body
/// that is no JSON object - cut off by max_tokens, or malformed - gives the header and the
/// body as it stands, for the client to refuse or repair.
pub fn parse_tool_call(header: &str, body: &str) -> (String, String) {
    let header = py_strip(header).to_string();
    match py_loads(body) {
        Ok(Value::Object(m)) => {
            let name = m.get("name").and_then(|n| n.as_str()).map(str::to_string).unwrap_or(header);
            let args = match m.get("args") {
                None | Some(Value::Null) => "{}".to_string(),
                Some(args) => py_json(args),
            };
            (name, args)
        }
        _ => (header, body.to_string()),
    }
}

/// The marker `s` begins with, and its length
fn mark_at(s: &str) -> Option<(usize, Mark)> {
    let fixed = [(MODEL, Mark::Model), (END_MESSAGE, Mark::End), (END_SAMPLING, Mark::Stop), (THINKING, Mark::Kind(Kind::Thinking)),
                 (TEXT, Mark::Kind(Kind::Text)), (INVOKE_TOOL_JSON, Mark::Kind(Kind::ToolJson))];
    fixed.iter().find(|(lit, _)| s.starts_with(lit)).map(|(lit, m)| (lit.len(), *m))
        .or_else(|| content_marker_len(s).map(|n| (n, Mark::Kind(Kind::Other))))
}

/// A whole answer's text, routed as InklingParser routes its ids. The text is the decode of
/// the ids (markers included); the one difference is a marker the model spelled out in
/// ordinary tokens, which is text to InklingParser and a marker here.
pub fn parse_answer(text: &str) -> Routed {
    let mut r = Routed::new();
    let mut sink = Vec::new();
    let (mut from, mut search) = (0, 0);
    while let Some(i) = text[search..].find("<|").map(|i| search + i) {
        match mark_at(&text[i..]) {
            Some((len, m)) => {
                r.text(&text[from..i], &mut sink);
                r.mark(m, &mut sink);
                from = i + len;
                search = from;
            }
            None => search = i + 2,
        }
    }
    r.text(&text[from..], &mut sink);
    r.close(&mut sink);
    r
}

/// (channel, text delta) events from the answer's ids, one id at a time: GlmParser's face
/// (feed, finish, tool_calls, the `reasoning` and `content` so far), so a server drives either.
///
/// A delta is the text of the ids since the last one, held back while it ends in U+FFFD (a
/// character split across ids), as GlmParser holds it: the tokenizer's decoder is ByteLevel
/// here too. Headers and tool-call bodies stream nothing.
pub struct InklingParser {
    ids: InklingIds,
    /// the current run's ids not yet released as text
    pending: Vec<u32>,
    routed: Routed,
}

impl Deref for InklingParser {
    type Target = Routed;

    fn deref(&self) -> &Routed {
        &self.routed
    }
}

impl InklingParser {
    pub fn new(ids: &InklingIds) -> Self {
        InklingParser { ids: ids.clone(), pending: Vec::new(), routed: Routed::new() }
    }

    fn flush(&mut self, text: &Text, last: bool, out: &mut Vec<(Channel, String)>) {
        if self.pending.is_empty() {
            return;
        }
        let delta = text.decode(&self.pending).unwrap_or_default();
        if !last && delta.ends_with('\u{FFFD}') {
            return;
        }
        self.pending.clear();
        self.routed.text(&delta, out);
    }

    pub fn feed(&mut self, text: &Text, tok: u32) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        if self.routed.ended {
            return out;
        }
        match self.ids.mark(tok) {
            Some(m) => {
                self.flush(text, true, &mut out);
                self.routed.mark(m, &mut out);
            }
            None => {
                self.pending.push(tok);
                self.flush(text, false, &mut out);
            }
        }
        out
    }

    /// The answer is over: what was held back, a split character and an open header too.
    pub fn finish(&mut self, text: &Text) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        self.flush(text, true, &mut out);
        self.routed.close(&mut out);
        out
    }

    /// (name, arguments JSON) of each tool call. GlmParser::tool_calls' signature; the JSON
    /// carries its own types, so neither the tokenizer nor the tools' schemas are consulted.
    pub fn tool_calls(&self, _text: &Text, _tools: Option<&Value>) -> Vec<(String, String)> {
        self.routed.tool_calls()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    // ------------------------------------------------------------ render
    //
    // The cases below are (messages, tools, reasoning_effort, result) with the result jinja2
    // 3.1.2 gave for the checkpoint's chat_template.jinja on 2026-11-16, in the environment
    // transformers' apply_chat_template builds:
    //
    //     env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
    //                                         extensions=["jinja2.ext.loopcontrols"])
    //     env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
    //         json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    //     env.globals["raise_exception"] = raise_exception          # raises TemplateError(msg)
    //     env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=True,
    //                                      reasoning_effort=effort)  # UNSET: not passed
    //
    // Renders has the rendered text, Raises a raise_exception message, and Fails is an error
    // inside jinja2 or json.dumps (the comment has it). Inputs are the JSON Python rendered.

    enum Want {
        Renders(&'static str),
        Raises(&'static str),
        Fails,
    }
    use Want::*;

    type Case = (&'static str, &'static str, &'static str, Want);

    fn run(messages: &str, tools: &str, effort: &str) -> Result<String, String> {
        let messages: Vec<Value> = serde_json::from_str(messages).unwrap();
        let tools: Value = serde_json::from_str(tools).unwrap();
        let effort: Option<Value> = (effort != "UNSET").then(|| serde_json::from_str(effort).unwrap());
        render_with_effort(&messages, Some(&tools), effort.as_ref())
    }

    fn check(cases: &[Case]) {
        for (messages, tools, effort, want) in cases {
            let got = run(messages, tools, effort);
            let got = got.as_ref().map(String::as_str).map_err(String::as_str);
            let at = format!("messages {messages}\ntools {tools}\neffort {effort}");
            match want {
                Renders(text) => assert_eq!(got, Ok(*text), "{at}"),
                Raises(message) => assert_eq!(got, Err(*message), "{at}"),
                Fails => assert!(got.is_err(), "{at}\nrendered {got:?}"),
            }
        }
    }

    #[test]
    fn plain_chats_and_system_prompts() {
        check(&[
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "system", "content": "A"}, {"role": "system", "content": "B"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>A<|end_message|><|message_system|><|content_text|>B<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"#)),
            (r#"[]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."}, {"role": "system", "content": "late system"}, {"role": "user", "content": "ünïcode ✓ <b> & 'q' \"d\"\n"}]"#, r#"null"#, r#"UNSET"#, Renders("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_text|>Hi.<|end_message|><|content_model_end_sampling|><|message_system|><|content_text|>late system<|end_message|><|message_user|><|content_text|>ünïcode ✓ <b> & 'q' \"d\"\n<|end_message|>")),
            (r#"[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": null}, {"role": "user"}, {"role": "system", "content": []}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|><|end_message|><|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>"#)),
        ]);
    }

    #[test]
    fn an_assistants_reasoning_is_its_own_message() {
        check(&[
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "The user greets me.", "content": "Hi there."}, {"role": "user", "content": "Bye"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_thinking|>The user greets me.<|end_message|><|message_model|><|content_text|>Hi there.<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>Bye<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "only thought", "content": ""}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_thinking|>only thought<|end_message|><|content_model_end_sampling|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "thought", "content": null}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_thinking|>thought<|end_message|><|content_model_end_sampling|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": "", "content": "x"}, {"role": "user", "reasoning_content": "ignored", "content": "y"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_text|>x<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>y<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": 5, "content": "x"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_thinking|>5<|end_message|><|message_model|><|content_text|>x<|end_message|><|content_model_end_sampling|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "reasoning_content": ["a", {"b": null, "c": 1.5, "d": true}, "it's"], "content": "x"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|><|content_thinking|>['a', {'b': None, 'c': 1.5, 'd': True}, "it's"]<|end_message|><|message_model|><|content_text|>x<|end_message|><|content_model_end_sampling|>"#)),
        ]);
    }

    #[test]
    fn tools_are_declared_as_sorted_compact_json() {
        check(&[
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]"#, r#"UNSET"#, Renders(r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather for a city","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]"#, r#"[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}, {"type": "function", "function": {"parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "a <b>query</b> & 'more' – ünïcode ✓\n\"quoted\"\t\u0001"}, "limit": {"type": "number", "default": 15000000000.0}, "frac": {"default": 0.1}, "tiny": {"default": 1e-07}, "whole": {"default": 100.0}, "flag": {"default": false}, "none": {"default": null}, "Z": {}, "a": {}, "é": {}}}, "name": "search", "strict": true}}, {"name": "flat_tool", "description": "no function wrapper", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}, {"type": "custom", "name": "odd", "description": 0, "parameters": []}]"#, r#"UNSET"#, Renders(r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather for a city","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"type":"object"},"type":"function"},{"description":"","name":"search","parameters":{"properties":{"Z":{},"a":{},"flag":{"default":false},"frac":{"default":0.1},"limit":{"default":15000000000.0,"type":"number"},"none":{"default":null},"q":{"description":"a <b>query</b> & 'more' – ünïcode ✓\n\"quoted\"\t\u0001","type":"string"},"tiny":{"default":1e-07},"whole":{"default":100.0},"é":{}},"type":"object"},"type":"function"},{"description":"no function wrapper","name":"flat_tool","parameters":{"properties":{"path":{"type":"string"}},"type":"object"},"type":"function"},{"description":"","name":"odd","parameters":{},"type":"custom"}]<|end_message|><|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"[]"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"[{"function": {"name": null}}, {"function": {"name": "n", "description": "", "parameters": {}}, "type": ""}]"#, r#"UNSET"#, Renders(r#"<|message_system|>tool_declare<|content_xml|>[{"description":"","name":null,"parameters":{},"type":"function"},{"description":"","name":"n","parameters":{},"type":"function"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"[{"type": "function", "function": {"description": "no name"}}]"#, r#"UNSET"#, Fails), // TypeError: Object of type Undefined is not JSON serializable
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"["text"]"#, r#"UNSET"#, Fails), // TypeError: Object of type Undefined is not JSON serializable
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"[{"function": null}]"#, r#"UNSET"#, Fails), // TypeError: Object of type Undefined is not JSON serializable
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"{"name": "f"}"#, r#"UNSET"#, Fails), // TypeError: Object of type Undefined is not JSON serializable
            (r#"[{"role": "user", "content": "Hello"}]"#, r#""abc""#, r#"UNSET"#, Fails), // TypeError: Object of type Undefined is not JSON serializable
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"5"#, r#"UNSET"#, Fails), // TypeError: 'int' object is not iterable
        ]);
    }

    #[test]
    fn tool_calls_and_their_results() {
        check(&[
            (r#"[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "Checking.", "reasoning_content": "think1", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 3, "city": "Paris"}}}, {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]}, {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, {"role": "tool", "tool_call_id": "c2", "content": "rainy"}, {"role": "assistant", "content": "Done."}, {"role": "user", "content": "Thanks"}]"#, r#"[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]"#, r#""low""#, Renders(r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather for a city","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|><|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Weather?<|end_message|><|message_model|><|content_thinking|>think1<|end_message|><|message_model|><|content_text|>Checking.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","days":3}}<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Rome"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|><|message_tool|>get_weather<|content_text|>rainy<|end_message|><|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>Thanks<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "search", "arguments": {"q": "x <y> & 'z' é", "n": [1, 2.5, null, true], "o": {"b": 1, "a": {}}}}}]}, {"role": "tool", "name": "search", "tool_call_id": "a", "content": "{\"hits\": 0}"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>search<|content_invoke_tool_json|>{"name":"search","args":{"n":[1,2.5,null,true],"o":{"a":{},"b":1},"q":"x <y> & 'z' é"}}<|end_message|><|content_model_end_sampling|><|message_tool|>search<|content_text|>{"hits": 0}<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": null}}, {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}}, {"id": "c", "type": "function", "function": {"name": "h", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "b", "content": "1"}, {"role": "tool", "tool_call_id": "zz", "content": "2"}, {"role": "tool", "content": "3"}, {"role": "tool", "tool_call_id": "c", "name": "", "content": [{"type": "text", "text": "4"}, "5"]}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}<|end_message|><|message_model|>g<|content_invoke_tool_json|>{"name":"g","args":{}}<|end_message|><|message_model|>h<|content_invoke_tool_json|>{"name":"h","args":{}}<|end_message|><|content_model_end_sampling|><|message_tool|>g<|content_text|>1<|end_message|><|message_tool|><|content_text|>2<|end_message|><|message_tool|><|content_text|>3<|end_message|><|message_tool|>h<|content_text|>4<|end_message|><|message_tool|>h<|content_text|>5<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "first", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "call_0", "content": "r1"}, {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "second", "arguments": {"k": 1}}}]}, {"role": "tool", "tool_call_id": "call_0", "content": "r2"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>first<|content_invoke_tool_json|>{"name":"first","args":{}}<|end_message|><|content_model_end_sampling|><|message_tool|>firstsecond<|content_text|>r1<|end_message|><|message_model|>second<|content_invoke_tool_json|>{"name":"second","args":{"k":1}}<|end_message|><|content_model_end_sampling|><|message_tool|>firstsecond<|content_text|>r2<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "name": "f", "content": "x", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>f<|content_text|>x<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":1}}<|end_message|><|content_model_end_sampling|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "name": "bot", "content": "x"}, {"role": "user", "name": "ignored", "content": "y"}, {"role": "tool", "name": 5, "content": "z"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>bot<|content_text|>x<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>y<|end_message|><|message_tool|>5<|content_text|>z<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}, {"id": 1, "function": {"name": "g"}}]}, {"role": "tool", "tool_call_id": 1.0, "content": "num id"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":1}}<|end_message|><|message_model|>g<|content_invoke_tool_json|>{"name":"g","args":{}}<|end_message|><|content_model_end_sampling|><|message_tool|>g<|content_text|>num id<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": []}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>"#)),
        ]);
    }

    #[test]
    fn tool_calls_the_template_refuses() {
        check(&[
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call arguments must be a parsed object, not a JSON string; canonicalize upstream"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": [1]}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call arguments must be an object"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": 5}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call arguments must be an object"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": 5, "arguments": {}}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call function name must be a string"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "function": {"arguments": {}}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call function name must be a string"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a", "function": "f"}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"tool call function name must be a string"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "name": "other", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {}}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"assistant name other does not match its tool call f"#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "a"}]}]"#, r#"null"#, r#"UNSET"#, Fails), // UndefinedError: 'dict object' has no attribute 'function'
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": ["f"]}]"#, r#"null"#, r#"UNSET"#, Fails), // UndefinedError: 'str object' has no attribute 'function'
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": {"id": "a"}}]"#, r#"null"#, r#"UNSET"#, Fails), // UndefinedError: 'str object' has no attribute 'function'
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": "abc"}]"#, r#"null"#, r#"UNSET"#, Fails), // UndefinedError: 'str object' has no attribute 'function'
            (r#"[{"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": 5}]"#, r#"null"#, r#"UNSET"#, Fails), // TypeError: 'int' object is not iterable
            (r#"[{"role": "tool", "tool_call_id": "a", "content": "r"}, {"role": "assistant", "tool_calls": [{"id": "a"}]}]"#, r#"null"#, r#"UNSET"#, Fails), // UndefinedError: 'dict object' has no attribute 'function'
            (r#"[{"role": "tool", "tool_call_id": "a", "content": "r"}, {"role": "assistant", "tool_calls": 5}]"#, r#"null"#, r#"UNSET"#, Fails), // TypeError: 'int' object is not iterable
        ]);
    }

    #[test]
    fn effort_levels() {
        check(&[
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""none""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""minimal""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""low""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""medium""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""high""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""xhigh""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""max""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"" low\n""#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"null"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0.0"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"-0.0"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0.5"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0.99"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0.123456789"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.123456789<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"1e-05"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 1e-05<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"false"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"[]"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"{"a": 1}"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"#)),
            (r#"[{"role": "system", "content": "S"}]"#, r#"null"#, r#""none""#, Renders(r#"<|message_system|><|content_text|>S<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"#)),
            (r#"[]"#, r#"null"#, r#"0.25"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.25<|end_message|>"#)),
        ]);
        check(&[
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""bogus""#, Raises(r#"Unknown reasoning_effort: bogus"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""""#, Raises(r#"Unknown reasoning_effort: "#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""0.5""#, Raises(r#"Unknown reasoning_effort: 0.5"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#""HIGH""#, Raises(r#"Unknown reasoning_effort: HIGH"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"1"#, Raises(r#"reasoning_effort must be in [0.0, 0.99]"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"0.991"#, Raises(r#"reasoning_effort must be in [0.0, 0.99]"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"-0.1"#, Raises(r#"reasoning_effort must be in [0.0, 0.99]"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"true"#, Raises(r#"reasoning_effort must be in [0.0, 0.99]"#)),
            (r#"[{"role": "user", "content": "Hello"}]"#, r#"null"#, r#"5"#, Raises(r#"reasoning_effort must be in [0.0, 0.99]"#)),
        ]);
    }

    #[test]
    fn content_parts() {
        check(&[
            (r#"[{"role": "user", "content": [{"type": "text", "text": "a"}, "b", {"type": "input_text", "text": "c"}, {"text": "d"}, {"type": "text"}, {"type": "text", "text": 5}, 7, null, ["x"]]}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>a<|end_message|><|message_user|><|content_text|>b<|end_message|><|message_user|><|content_text|>c<|end_message|><|message_user|><|content_text|>d<|end_message|><|message_user|><|content_text|><|end_message|><|message_user|><|content_text|><|end_message|><|message_user|><|content_text|><|end_message|><|message_user|><|content_text|><|end_message|><|message_user|><|content_text|><|end_message|>"#)),
            (r#"[{"role": "user", "content": [{"type": "text", "text": "Look:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "image"}, {"type": "input_image", "image_url": "x"}]}, {"role": "tool", "name": "shot", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look:<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_tool|>shot<|content_image|><|unused_200054|><|end_message|>"#)),
            (r#"[{"role": "user", "content": {"k1": 1, "k2": 2}}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>k1<|end_message|><|message_user|><|content_text|>k2<|end_message|>"#)),
            (r#"[{"role": "system", "content": [{"type": "text", "text": "sys part"}]}, {"role": "assistant", "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}], "reasoning_content": "r"}]"#, r#"null"#, r#"UNSET"#, Renders(r#"<|message_system|><|content_text|>sys part<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_model|><|content_thinking|>r<|end_message|><|message_model|><|content_text|>p1<|end_message|><|message_model|><|content_text|>p2<|end_message|><|content_model_end_sampling|>"#)),
        ]);
        check(&[
            (r#"[{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Image content is only supported for user and tool messages"#)),
            (r#"[{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Image content is only supported for user and tool messages"#)),
            (r#"[{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Inkling 2 does not accept audio input"#)),
            (r#"[{"role": "user", "content": [{"type": "audio"}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Inkling 2 does not accept audio input"#)),
            (r#"[{"role": "user", "content": [{"type": "audio_url"}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Inkling 2 does not accept audio input"#)),
            (r#"[{"role": "user", "content": [{"type": "video"}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unsupported content part type: video"#)),
            (r#"[{"role": "user", "content": [{"type": null}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unsupported content part type: None"#)),
            (r#"[{"role": "user", "content": [{"type": 5}]}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unsupported content part type: 5"#)),
            (r#"[{"role": "user", "content": 5}]"#, r#"null"#, r#"UNSET"#, Fails), // TypeError: 'int' object is not iterable
            (r#"[{"role": "user", "content": true}]"#, r#"null"#, r#"UNSET"#, Fails), // TypeError: 'bool' object is not iterable
        ]);
    }

    #[test]
    fn unknown_roles_are_refused() {
        check(&[
            (r#"[{"role": "developer", "content": "x"}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: developer"#)),
            (r#"[{"content": "x"}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: "#)),
            (r#"[{"role": null, "content": "x"}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: None"#)),
            (r#"[{"role": 5, "content": "x"}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: 5"#)),
            (r#"["text"]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: "#)),
            (r#"[{"role": "user", "content": "Hello"}, {"role": "function", "content": "x"}]"#, r#"null"#, r#"UNSET"#, Raises(r#"Unknown message role: function"#)),
            (r#"[{"role": ["user"], "content": "x"}]"#, r#"null"#, r#"UNSET"#, Fails), // TypeError: unhashable type: 'list'
        ]);
    }

    #[test]
    fn render_takes_an_effort_by_name() {
        let msgs = [json!({"role": "user", "content": "Hello"})];
        let at = |level: &str| format!("{SYSTEM}{TEXT}Thinking effort level: {level}{END_MESSAGE}{USER}{TEXT}Hello{END_MESSAGE}");
        for (name, level) in [("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99")] {
            assert_eq!(render(&msgs, None, Some(name)), Ok(at(level)), "{name}");
            assert_eq!(effort_level(name).map(|l| if l == 0.0 { "0".to_string() } else { py_float(l) }).as_deref(), Some(level));
        }
        assert_eq!(render(&msgs, None, None), Ok(at("0.9")));
        assert_eq!(render(&msgs, Some(&Value::Null), None), Ok(at("0.9")));
        // a number is not a name: the template refuses the string
        assert_eq!(render(&msgs, None, Some("0.5")), Err("Unknown reasoning_effort: 0.5".into()));
        assert_eq!(render_with_effort(&msgs, None, Some(&json!(0.5))), Ok(at("0.5")));
    }

    #[test]
    fn sorted_json_is_json_dumps_sorted_and_compact() {
        // json.dumps({...}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), Python 3
        let v: Value = serde_json::from_str(r#"{"b": [1, 2.0, 1e-7, 1.5e300, -0.0, true, null], "a": {"é": "<&'>\u2028\"\\\n", "Z": {}, "a": []}, "": 0}"#).unwrap();
        assert_eq!(sorted_json(&v), "{\"\":0,\"a\":{\"Z\":{},\"a\":[],\"é\":\"<&'>\u{2028}\\\"\\\\\\n\"},\"b\":[1,2.0,1e-07,1.5e+300,-0.0,true,null]}");
    }

    #[test]
    fn prepare_parses_the_arguments_an_openai_client_sends_as_strings() {
        let call = |args: Value| json!({"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}});
        let msgs = vec![json!({"role": "user", "content": "Hello"}),
                        json!({"role": "assistant", "content": null,
                               "tool_calls": [call(json!("{\"b\": 1, \"a\": \"x\"}")), call(json!("")), call(json!("{\"cut\": ")), call(json!({"k": 1}))]}),
                        json!({"role": "tool", "tool_call_id": "c1", "content": "r"})];
        // as sent, the template refuses it
        assert_eq!(render(&msgs, None, None), Err("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream".into()));
        let ready = prepare(&msgs);
        assert!(matches!(ready, Cow::Owned(_)));
        let want = format!("{SYSTEM}{TEXT}Thinking effort level: 0.9{END_MESSAGE}{USER}{TEXT}Hello{END_MESSAGE}\
                            {MODEL}f{INVOKE_TOOL_JSON}{{\"name\":\"f\",\"args\":{{\"a\":\"x\",\"b\":1}}}}{END_MESSAGE}\
                            {MODEL}f{INVOKE_TOOL_JSON}{{\"name\":\"f\",\"args\":{{}}}}{END_MESSAGE}\
                            {MODEL}f{INVOKE_TOOL_JSON}{{\"name\":\"f\",\"args\":{{\"input\":\"{{\\\"cut\\\": \"}}}}{END_MESSAGE}\
                            {MODEL}f{INVOKE_TOOL_JSON}{{\"name\":\"f\",\"args\":{{\"k\":1}}}}{END_MESSAGE}\
                            {END_SAMPLING}{TOOL}ffff{TEXT}r{END_MESSAGE}"); // all four calls have the result's id
        assert_eq!(render(&ready, None, None), Ok(want));
        // nothing to parse: the caller's messages, not a copy; and shapes prepare does not know pass through
        assert!(matches!(prepare(&msgs[..1]), Cow::Borrowed(_)));
        let odd = [json!("text"), json!({"role": "assistant", "tool_calls": 5}), json!({"role": "assistant", "tool_calls": [5, {"function": "f"}]})];
        assert!(matches!(prepare(&odd), Cow::Borrowed(_)));
    }

    #[test]
    fn image_parts_are_counted_as_render_writes_them() {
        let img = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}});
        let msgs = [json!({"role": "user", "content": [{"type": "text", "text": "Look:"}, img, {"type": "input_image"}, "image"]}),
                    json!({"role": "tool", "name": "shot", "content": [{"type": "image"}]}),
                    json!({"role": "user", "content": "image_url"})];
        assert_eq!(image_parts(&msgs), 3);
        assert_eq!(render(&msgs, None, None).unwrap().matches(IMAGE_PLACEHOLDER).count(), 3);
        assert_eq!(image_parts(&msgs[2..]), 0);
    }

    // ------------------------------------------------------------ the answer

    /// (the answer's text, reasoning, content, tool calls as (name, arguments), ended)
    type Answer = (&'static str, &'static str, &'static str, &'static [(&'static str, &'static str)], bool);

    const ANSWERS: &[Answer] = &[
        // thinking, then text
        ("<|message_model|><|content_thinking|>The user greets me.\nBe brief.<|end_message|><|message_model|><|content_text|>Hello! How can I help?<|end_message|><|content_model_end_sampling|>",
         "The user greets me.\nBe brief.", "Hello! How can I help?", &[], true),
        // text only
        ("<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>", "", "Hello!", &[], true),
        // one tool call
        ("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|><|content_model_end_sampling|>",
         "", "", &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}")], true),
        // thinking, text and two tool calls
        ("<|message_model|><|content_thinking|>I need both cities.<|end_message|><|message_model|><|content_text|>Checking.<|end_message|>\
          <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|>\
          <|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{\"q\":\"Rome – météo ✓\",\"n\":[1,2.5,null],\"deep\":{\"a\":true}}}<|end_message|><|content_model_end_sampling|>",
         "I need both cities.", "Checking.",
         &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}"), ("search", "{\"q\": \"Rome – météo ✓\", \"n\": [1, 2.5, null], \"deep\": {\"a\": true}}")], true),
        // several messages of one kind join without a separator
        ("<|message_model|><|content_thinking|>a<|end_message|><|message_model|><|content_thinking|>b<|end_message|><|message_model|><|content_text|>c<|end_message|><|message_model|><|content_text|>d<|end_message|><|content_model_end_sampling|>",
         "ab", "cd", &[], true),
        // cut off (max_tokens) in the reasoning, in the text, in a tool call's body and in its header
        ("<|message_model|><|content_thinking|>The user wants", "The user wants", "", &[], false),
        ("<|message_model|><|content_thinking|>t<|end_message|><|message_model|><|content_text|>Hello, wor", "t", "Hello, wor", &[], false),
        ("<|message_model|><|content_text|>On it.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Par",
         "", "On it.", &[("get_weather", "{\"name\":\"get_weather\",\"args\":{\"city\":\"Par")], false),
        ("<|message_model|><|content_text|>On it.<|end_message|><|message_model|>get_wea", "", "On it.get_wea", &[], false),
        // other content kinds are content, their header dropped; a call's name falls back to its header
        ("<|message_model|><|content_xml|><a b=\"1\"/><|end_message|><|message_model|>patch<|content_invoke_tool_text|>raw text<|end_message|>\
          <|message_model|><|content_tool_error|>no such tool<|end_message|><|message_model|> f <|content_invoke_tool_json|>{\"args\":{\"x\":1}}<|end_message|>\
          <|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\"}<|end_message|><|message_model|>h<|content_invoke_tool_json|>[1]<|end_message|><|content_model_end_sampling|>",
         "", "<a b=\"1\"/>raw textno such tool", &[("f", "{\"x\": 1}"), ("g", "{}"), ("h", "[1]")], true),
        // a missing <|end_message|>, a missing <|message_model|>, text between messages, an empty message, a
        // header without a content token; nothing after the stop
        ("stray<|content_thinking|>r1<|message_model|><|content_text|>c1<|content_thinking|>r2<|end_message|> between <|end_message|>\
          <|message_model|><|content_text|><|end_message|><|message_model|>lost<|end_message|><|message_model|>lost2<|message_model|><|content_text|>c2\
          <|content_model_end_sampling|><|message_model|><|content_text|>after",
         "r1r2", "strayc1 between lostlost2c2", &[], true),
        // a marker-like text that is no marker
        ("<|message_model|><|content_text|>a <| b <|content_Text|> c <|content_|> d <|end_message |><|end_message|><|content_model_end_sampling|>",
         "", "a <| b <|content_Text|> c <|content_|> d <|end_message |>", &[], true),
    ];

    fn check_routed(r: &Routed, a: &Answer) {
        let calls: Vec<(String, String)> = a.3.iter().map(|(n, args)| (n.to_string(), args.to_string())).collect();
        assert_eq!((r.reasoning.as_str(), r.content.as_str(), r.tool_calls(), r.ended), (a.1, a.2, calls, a.4), "{}", a.0);
    }

    #[test]
    fn a_whole_answer_is_routed_by_its_markers() {
        for a in ANSWERS {
            check_routed(&parse_answer(a.0), a);
        }
        let r = parse_answer(ANSWERS[3].0);
        assert_eq!(r.calls[0], RawCall { header: "get_weather".into(), json: "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}".into(), closed: true });
        assert_eq!(r.call_texts().len(), 2);
        // a call the answer was cut inside is reported, and marked
        assert!(!parse_answer(ANSWERS[7].0).calls[0].closed);
    }

    #[test]
    fn tool_call_bodies() {
        assert_eq!(parse_tool_call("f", r#"{"name":"g","args":{"b":1,"a":[-0, 1.0, "é"]}}"#), ("g".into(), r#"{"b": 1, "a": [0, 1.0, "é"]}"#.into()));
        assert_eq!(parse_tool_call("\n f\t", r#" {"name": 5, "args": null} "#), ("f".into(), "{}".into()));
        assert_eq!(parse_tool_call("f", r#"{"name":"f","args":"text"}"#), ("f".into(), r#""text""#.into()));
        assert_eq!(parse_tool_call("f", "not json"), ("f".into(), "not json".into()));
        assert_eq!(parse_tool_call("", ""), (String::new(), String::new()));
    }

    fn model_dir() -> Option<PathBuf> {
        let dir = std::env::var("INKLING_TOKENIZER_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join("models/Inkling-2-Small-EXL3-4bpw")
        });
        dir.join("tokenizer.json").exists().then_some(dir)
    }

    /// The checkpoint's tokenizer and ids, loaded once for every test here (a 26 MB file)
    fn tokenizer() -> Option<&'static (Text, InklingIds)> {
        static LOADED: OnceLock<Option<(Text, InklingIds)>> = OnceLock::new();
        let loaded = LOADED.get_or_init(|| {
            let dir = model_dir()?;
            let text = Text::load(&dir).unwrap();
            let ids = InklingIds::load(&dir, &text).unwrap();
            Some((text, ids))
        });
        if loaded.is_none() {
            eprintln!("Inkling tokenizer not present; skipping");
        }
        loaded.as_ref()
    }

    #[test]
    fn the_special_tokens_are_single_ids() {
        let Some((text, ids)) = tokenizer() else { return };
        let specials = [(ENDOFTEXT, 199999), (USER, 200000), (MODEL, 200001), (SYSTEM, 200002), (TOOL, 200003), (TEXT, 200004), (IMAGE, 200005),
                        (END_SAMPLING, 200006), (THINKING, 200008), (END_MESSAGE, 200010), (TOOL_ERROR, 200022), (XML, 200024),
                        (BEGIN_OF_TEXT, 200028), (INVOKE_TOOL_JSON, 200049), (IMAGE_PLACEHOLDER, 200054), (INVOKE_TOOL_TEXT, 200057)];
        for (token, id) in specials {
            assert_eq!(text.encode(token).unwrap(), vec![id], "{token}");
            assert_eq!(text.decode(&[id]).unwrap(), token);
            // and inside text, with nothing added around it
            assert_eq!(text.encode(&format!("a{token}b")).unwrap(), [text.encode("a").unwrap(), vec![id], text.encode("b").unwrap()].concat(), "{token}");
        }
        assert_eq!(*ids, InklingIds { model: 200001, end: 200010, thinking: 200008, text: 200004, tool_json: 200049,
                                      other: vec![200005, 200022, 200024, 200057], image: 200054, stop: vec![200006] });
    }

    #[test]
    fn a_rendered_prompt_round_trips_through_the_tokenizer() {
        let Some((text, _)) = tokenizer() else { return };
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Get the weather – ünïcode ✓",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        let msgs = [json!({"role": "system", "content": "Be brief."}),
                    json!({"role": "user", "content": [{"type": "text", "text": "Weather in Paris?\n"}, {"type": "image"}]}),
                    json!({"role": "assistant", "reasoning_content": "I should look.", "content": "Checking.",
                           "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}),
                    json!({"role": "tool", "tool_call_id": "c1", "content": "sunny, 21°C"})];
        let prompt = render(&msgs, Some(&tools), Some("low")).unwrap();
        let ids = text.encode(&prompt).unwrap();
        assert_eq!(text.decode(&ids).unwrap(), prompt);
        // no begin-of-text and nothing appended: the template's first and last markers
        assert_eq!((ids[0], ids[ids.len() - 1]), (200002, 200010));
        let count = |id: u32| ids.iter().filter(|&&t| t == id).count();
        // system: tools, the prompt, the effort; user: text and image; model: thinking, text, the call; tool: the result
        assert_eq!([count(200002), count(200000), count(200001), count(200003), count(200010), count(200006), count(200054)], [3, 2, 3, 1, 9, 1, 1]);
        assert_eq!([count(200024), count(200004), count(200005), count(200008), count(200049)], [1, 5, 1, 1, 1]);
        // every marker in the text is one of those ids: none was split into ordinary tokens
        assert_eq!(prompt.matches("<|").count(), ids.iter().filter(|&&t| t >= 199998).count());
    }

    fn feed_all(text: &Text, ids: &InklingIds, toks: &[u32]) -> (InklingParser, Vec<(Channel, String)>) {
        let mut p = InklingParser::new(ids);
        let mut events = Vec::new();
        for &t in toks {
            events.extend(p.feed(text, t));
        }
        events.extend(p.finish(text));
        (p, events)
    }

    fn joined(events: &[(Channel, String)], chan: Channel) -> String {
        events.iter().filter(|(c, _)| *c == chan).map(|(_, s)| s.as_str()).collect()
    }

    #[test]
    fn ids_fed_one_at_a_time_route_as_the_whole_text_does() {
        let Some((text, ids)) = tokenizer() else { return };
        for a in ANSWERS {
            let toks = text.encode(a.0).unwrap();
            let (p, events) = feed_all(text, ids, &toks);
            check_routed(&p, a);
            let whole = parse_answer(&text.decode(&toks).unwrap());
            assert_eq!((&p.reasoning, &p.content, &p.calls, p.ended), (&whole.reasoning, &whole.content, &whole.calls, whole.ended), "{}", a.0);
            // the deltas are the texts, in pieces, and none is empty
            assert_eq!((joined(&events, Channel::Reasoning), joined(&events, Channel::Content)), (p.reasoning.clone(), p.content.clone()), "{}", a.0);
            assert!(events.iter().all(|(_, s)| !s.is_empty()));
            assert_eq!(p.tool_calls(text, None), whole.tool_calls());
        }
    }

    #[test]
    fn deltas_stream_as_the_ids_arrive() {
        let Some((text, ids)) = tokenizer() else { return };
        let toks = text.encode("<|message_model|><|content_thinking|>The user greets me.<|end_message|><|message_model|><|content_text|>Hello there, friend!<|end_message|><|content_model_end_sampling|><|message_model|><|content_text|>more").unwrap();
        let mut p = InklingParser::new(ids);
        let mut events = Vec::new();
        for &t in &toks {
            let ev = p.feed(text, t);
            // a marker says nothing; a text id is its own delta, at once
            assert_eq!(ev.len(), usize::from(t < 199998 && !p.ended), "{t}");
            events.extend(ev);
        }
        assert!(p.ended && p.finish(text).is_empty());
        let thought = text.encode("The user greets me.").unwrap().len();
        let said = text.encode("Hello there, friend!").unwrap().len();
        assert!(thought > 2 && said > 2);
        assert_eq!(events.iter().map(|(c, _)| *c).collect::<Vec<_>>(), [vec![Channel::Reasoning; thought], vec![Channel::Content; said]].concat());
        assert_eq!((p.reasoning.as_str(), p.content.as_str()), ("The user greets me.", "Hello there, friend!"));
    }

    #[test]
    fn a_character_split_across_ids_is_held_until_it_is_whole() {
        let Some((text, ids)) = tokenizer() else { return };
        // a character the vocabulary has no single token for
        let split = ["𓀀", "🫠", "ꙮ", "𝕏", "\u{10FFFD}", "ᚠ"].into_iter().find(|c| {
            let t = text.encode(c).unwrap();
            t.len() > 1 && text.decode(&t[..1]).unwrap().ends_with('\u{FFFD}')
        });
        let Some(c) = split else {
            eprintln!("no candidate character splits across ids; skipping");
            return;
        };
        let answer = format!("<|message_model|><|content_text|>a{c}b<|end_message|><|message_model|><|content_thinking|>{c}");
        let toks = text.encode(&answer).unwrap();
        let (p, events) = feed_all(text, ids, &toks);
        assert!(events.iter().all(|(_, s)| !s.contains('\u{FFFD}')), "{events:?}");
        assert_eq!((p.content.as_str(), p.reasoning.as_str()), (format!("a{c}b").as_str(), c));
        // cut inside the character: finish releases what there is
        let (p, events) = feed_all(text, ids, &toks[..toks.len() - 1]);
        assert_eq!(events.last().map(|(ch, s)| (*ch, s.ends_with('\u{FFFD}'))), Some((Channel::Reasoning, true)));
        assert_eq!(p.content, format!("a{c}b"));
    }
}
