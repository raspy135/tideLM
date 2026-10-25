//! GLM-5.3's chat prompt and the router of its answers: server/glm_app.py's GlmTemplate and
//! GlmParser by hand.
//!
//! render is what jinja2 makes of the checkpoint's chat_template.jinja with glm_app's settings
//! (trim_blocks, lstrip_blocks, a `tojson` that is Python's json.dumps): `[gMASK]<sop>`, the
//! effort line (low / high, anything else Max), the tool list, then the turns, ending in the
//! generation prompt `<|assistant|><think>`. glm_app's own step comes first: a tool call's
//! `arguments` string is JSON-parsed (bad JSON becomes `{"input": raw}`). The template's
//! value rules are kept where they decide bytes - jinja2's Undefined for a missing key, Python
//! truthiness, `str()` of a non-string - and where it raises, render refuses.
//!
//! GlmParser routes the answer's token ids: reasoning until `</think>` (the prompt ends inside
//! `<think>`), content after it, `<tool_call>`...`</tool_call>` collected as ids and parsed
//! when the answer is done; stop ids are dropped. Every marker is a single token.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::chat::{py_json, py_loads, py_number};
use crate::{Error, Text};

pub const PREFIX: &str = "[gMASK]<sop>";
pub const SYSTEM: &str = "<|system|>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";
pub const OBSERVATION: &str = "<|observation|>";
pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const THINK: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOL_RESPONSE: &str = "<tool_response>";
pub const TOOL_RESPONSE_END: &str = "</tool_response>";
pub const ARG_KEY: &str = "<arg_key>";
pub const ARG_KEY_END: &str = "</arg_key>";
pub const ARG_VALUE: &str = "<arg_value>";
pub const ARG_VALUE_END: &str = "</arg_value>";
/// the end of every prompt: generation starts inside the reasoning block
pub const GENERATION_PROMPT: &str = "<|assistant|><think>";

const TOOLS_HEAD: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_TAIL: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n\
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
const MEDIA: [&str; 7] = ["image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"];

// ---------------------------------------------------------------- Python's values

/// `str.isspace()` (also what `\s` and `str.strip()` use): Rust's White_Space plus the four
/// information separators U+001C..U+001F.
pub fn py_isspace(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\u{1c}'..='\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}'
                | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// `str.strip()`
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// Python truthiness of a JSON value
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Characters `repr()` writes escaped. Unicode's printable test without its tables: the C0/C1
/// controls, the space separators and line/paragraph separators, the common format
/// characters and private use.
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

/// `repr()` of a JSON value as Python holds it (dict, list, str, int, float, bool, None)
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

/// What `{{ v }}` writes: `str(v)`
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => py_number(n),
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

// ---------------------------------------------------------------- jinja2's values

/// A template expression's value: a JSON value, jinja2's Undefined (a missing key or
/// element), or a defined value that has no keys (a string's first character).
#[derive(Clone, Copy)]
enum J<'a> {
    Undef,
    Other,
    V(&'a Value),
}

fn undefined(what: &str) -> String {
    format!("UndefinedError: {what}")
}

impl<'a> J<'a> {
    fn of(v: Option<&'a Value>) -> Self {
        v.map(J::V).unwrap_or(J::Undef)
    }

    /// `x.key`: a mapping's item, Undefined for anything else; Undefined's raises
    fn attr(self, k: &str) -> Result<J<'a>, String> {
        match self {
            J::Undef => Err(undefined(&format!("an undefined value has no attribute '{k}'"))),
            J::V(Value::Object(m)) => Ok(J::of(m.get(k))),
            _ => Ok(J::Undef),
        }
    }

    /// `x.0`
    fn first(self) -> Result<J<'a>, String> {
        match self {
            J::Undef => Err(undefined("an undefined value has no element 0")),
            J::V(Value::Array(a)) => Ok(J::of(a.first())),
            J::V(Value::String(s)) if !s.is_empty() => Ok(J::Other),
            _ => Ok(J::Undef),
        }
    }

    fn truthy(self) -> bool {
        match self {
            J::Undef => false,
            J::Other => true,
            J::V(v) => truthy(v),
        }
    }

    fn defined(self) -> bool {
        !matches!(self, J::Undef)
    }

    fn string(self) -> Option<&'a str> {
        match self {
            J::V(Value::String(s)) => Some(s),
            _ => None,
        }
    }

    fn is(self, s: &str) -> bool {
        self.string() == Some(s)
    }

    fn mapping(self) -> bool {
        matches!(self, J::V(Value::Object(_)))
    }

    /// `is iterable` (Undefined iterates as empty)
    fn iterable(self) -> bool {
        matches!(self, J::Undef | J::Other | J::V(Value::String(_) | Value::Array(_) | Value::Object(_)))
    }

    /// `{{ x }}`
    fn render(self) -> String {
        match self {
            J::V(v) => py_str(v),
            _ => String::new(),
        }
    }

    /// `==`: two Undefined are equal
    fn eq(self, other: J) -> bool {
        match (self, other) {
            (J::Undef, J::Undef) => true,
            (J::V(a), J::V(b)) => py_eq(a, b),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------- the template's macros

/// The macro `visible_text`: strings and text parts, a reminder for each media part
fn visible_text(content: J) -> String {
    match content {
        J::V(Value::String(s)) => s.clone(),
        J::V(Value::Array(items)) => {
            let mut out = String::new();
            for item in items {
                match item {
                    Value::Object(m) => {
                        let ty = J::of(m.get("type"));
                        if ty.is("text") {
                            out.push_str(&J::of(m.get("text")).render());
                        } else if let Some(t) = ty.string().filter(|t| MEDIA.contains(t)) {
                            let media = t.replace("_url", "").replace("input_", "");
                            out.push_str(&format!("<reminder>You are unable to process this {media} because you don't have \
                                                   multi-modal input ability. Try different methods.</reminder>"));
                        }
                    }
                    Value::String(s) => out.push_str(s),
                    _ => {}
                }
            }
            out
        }
        J::Undef | J::V(Value::Null) => String::new(),
        other => other.render(),
    }
}

/// The macro `tool_to_json`: the definition's keys in the client's order, without
/// defer_loading and strict, each value json.dumps'ed (the key itself is written raw)
fn tool_to_json(tool: J) -> Result<String, String> {
    let J::V(Value::Object(m)) = tool else {
        return Err(undefined("a tool definition has no attribute 'items'"));
    };
    let items: Vec<String> = m
        .iter()
        .filter(|(k, _)| *k != "defer_loading" && *k != "strict")
        .map(|(k, v)| format!("\"{k}\": {}", py_json(v)))
        .collect();
    Ok(format!("{{{}}}", items.join(", ")))
}

/// `{%- if 'function' in tool -%}{%- set tool = tool['function'] -%}`
fn unwrap_tool(tool: &Value) -> Result<J<'_>, String> {
    match tool {
        Value::Object(m) => Ok(m.get("function").map(J::V).unwrap_or(J::V(tool))),
        _ => Err(format!("TypeError: a tool must be an object, not {}", py_repr(tool))),
    }
}

struct Tpl<'a> {
    msgs: &'a [Value],
    /// truthy: the tool list was rendered and its macros are defined
    tools: Option<&'a [Value]>,
}

fn id_of(obj: J) -> String {
    let J::V(Value::Object(m)) = obj else {
        return String::new();
    };
    for k in ["tool_call_id", "id"] {
        let v = J::of(m.get(k));
        if v.truthy() {
            return v.render();
        }
    }
    String::new()
}

fn content(m: &Value) -> J<'_> {
    J::of(m.get("content"))
}

fn role(m: &Value) -> J<'_> {
    J::of(m.get("role"))
}

/// `x is iterable and x is not string and x is not mapping and x and x.0.type == "tool_reference"`
fn is_references(x: J) -> Result<bool, String> {
    Ok(x.iterable() && x.string().is_none() && !x.mapping() && x.truthy() && x.first()?.attr("type")?.is("tool_reference"))
}

impl<'a> Tpl<'a> {
    /// The macro `tool_references_to_response`: each referenced tool's definition, deferred
    /// ones too
    fn references(&self, refs: J) -> Result<String, String> {
        let Some(tools) = self.tools else {
            return Err(undefined("'tool_references_to_response' is undefined"));
        };
        let mut out = String::from("<tool_response><tools>\n");
        if let J::V(Value::Array(refs)) = refs {
            for tr in refs {
                let want = J::V(tr).attr("name")?;
                for tool in tools {
                    let tool = unwrap_tool(tool)?;
                    if tool.attr("name")?.eq(want) {
                        out.push_str(&tool_to_json(tool)?);
                        out.push('\n');
                    }
                }
            }
        }
        out.push_str("</tools></tool_response>");
        Ok(out)
    }

    /// The macro `is_list_of_outputs`
    fn list_of_outputs(m: &Value) -> Result<bool, String> {
        let c = content(m);
        Ok(c.truthy() && c.first()?.attr("output")?.defined())
    }

    fn output(&self, out: J) -> Result<String, String> {
        if is_references(out)? {
            self.references(out)
        } else {
            Ok(format!("{TOOL_RESPONSE}{}{TOOL_RESPONSE_END}", visible_text(out)))
        }
    }

    /// The macro `render_tool_response`
    fn tool_response(&self, m: &Value) -> Result<String, String> {
        let c = content(m);
        if let Some(s) = c.string() {
            return Ok(format!("{TOOL_RESPONSE}{s}{TOOL_RESPONSE_END}"));
        }
        if c.truthy() && !c.mapping() && c.first()?.attr("type")?.is("tool_reference") {
            return self.references(c);
        }
        if Self::list_of_outputs(m)? {
            let mut out = String::new();
            if let J::V(Value::Array(entries)) = c {
                for tr in entries {
                    out.push_str(&self.output(J::V(tr).attr("output")?)?);
                }
            }
            return Ok(out);
        }
        Ok(format!("{TOOL_RESPONSE}{}{TOOL_RESPONSE_END}", visible_text(c)))
    }

    /// Each tool result's id in messages[lo..=hi], list-of-outputs entries one by one
    fn result_ids(&self, lo: usize, hi: usize) -> Result<Vec<Vec<String>>, String> {
        self.msgs[lo..=hi]
            .iter()
            .map(|m| {
                Ok(match (Self::list_of_outputs(m)?, content(m)) {
                    (true, J::V(Value::Array(entries))) => entries.iter().map(|e| id_of(J::V(e))).collect(),
                    _ => vec![id_of(J::V(m))],
                })
            })
            .collect()
    }

    /// The run of tool messages from `start`: one `<|observation|>`, the results in the order
    /// of the assistant's calls when every result's id is unique and among them (and the calls'
    /// ids are unique), else as sent
    fn tool_block(&self, start: usize) -> Result<String, String> {
        let msgs = self.msgs;
        let end = (start..msgs.len()).take_while(|&j| role(&msgs[j]).is("tool")).last().unwrap_or(start);
        let calls: Option<&Vec<Value>> = match start.checked_sub(1).map(|p| &msgs[p]) {
            Some(prev) if role(prev).is("assistant") && J::of(prev.get("tool_calls")).truthy() => prev["tool_calls"].as_array(),
            _ => None,
        };
        let mut out = String::from(OBSERVATION);
        let sorted = match calls {
            None => false,
            Some(calls) => {
                let ids = self.result_ids(start, end)?;
                let all: Vec<&String> = ids.iter().flatten().collect();
                let call_ids: Vec<String> = calls.iter().map(|c| id_of(J::V(c))).collect();
                let results_ok = all.iter().all(|id| !id.is_empty() && all.iter().filter(|x| x == &id).count() < 2 && call_ids.contains(id));
                let calls_ok = call_ids.iter().enumerate().all(|(i, id)| !id.is_empty() && !call_ids[i + 1..].contains(id));
                if results_ok && calls_ok {
                    for want in &call_ids {
                        for m in &msgs[start..=end] {
                            match (Self::list_of_outputs(m)?, content(m)) {
                                (true, J::V(Value::Array(entries))) => {
                                    for e in entries {
                                        if id_of(J::V(e)) == *want {
                                            out.push_str(&self.output(J::V(e).attr("output")?)?);
                                        }
                                    }
                                }
                                _ => {
                                    if id_of(J::V(m)) == *want {
                                        out.push_str(&self.tool_response(m)?);
                                    }
                                }
                            }
                        }
                    }
                }
                results_ok && calls_ok
            }
        };
        if !sorted {
            for m in &msgs[start..=end] {
                out.push_str(&self.tool_response(m)?);
            }
        }
        Ok(out)
    }

    fn assistant(&self, m: &Value) -> Result<String, String> {
        let mut text = visible_text(content(m));
        let reasoning = match m.get("reasoning_content") {
            Some(Value::String(r)) => Some(r.clone()),
            _ if text.contains(THINK_END) => {
                let r = text.split(THINK_END).next().unwrap_or("").rsplit(THINK).next().unwrap_or("").to_string();
                text = text.rsplit(THINK_END).next().unwrap_or("").to_string();
                Some(r)
            }
            _ => None,
        };
        let mut out = String::from(ASSISTANT);
        out.push_str(THINK);
        out.push_str(reasoning.as_deref().unwrap_or(""));
        out.push_str(THINK_END);
        out.push_str(py_strip(&text));
        let calls = J::of(m.get("tool_calls"));
        if calls.truthy() {
            let Some(calls) = m["tool_calls"].as_array() else {
                return Err("TypeError: tool_calls must be a list".into());
            };
            for tc in calls {
                let mut tc = J::V(tc);
                let f = tc.attr("function")?;
                if f.truthy() {
                    tc = f;
                }
                let Some(name) = tc.attr("name")?.string() else {
                    return Err(undefined("a tool call has no attribute 'name' (or it is not a string)"));
                };
                out.push_str(TOOL_CALL);
                out.push_str(name);
                let J::V(Value::Object(args)) = tc.attr("arguments")? else {
                    return Err(undefined("a tool call's arguments have no attribute 'items'"));
                };
                for (k, v) in args {
                    out.push_str(ARG_KEY);
                    out.push_str(k);
                    out.push_str(ARG_KEY_END);
                    out.push_str(ARG_VALUE);
                    match v {
                        Value::String(s) => out.push_str(s),
                        v => out.push_str(&py_json(v)),
                    }
                    out.push_str(ARG_VALUE_END);
                }
                out.push_str(TOOL_CALL_END);
            }
        }
        Ok(out)
    }
}

/// glm_app's step before the template: each message a copy, each tool call's `arguments`
/// string parsed (empty: `{}`; not JSON: `{"input": raw}`).
fn prepare(messages: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let Value::Object(m) = m else {
            return Err(format!("ValueError: messages[{i}] must be an object"));
        };
        let mut m = m.clone();
        if m.get("tool_calls").is_some_and(truthy) {
            let Some(calls) = m["tool_calls"].as_array() else {
                return Err(format!("AttributeError: messages[{i}].tool_calls must be a list of objects"));
            };
            let mut parsed = Vec::with_capacity(calls.len());
            for tc in calls {
                let Value::Object(tc) = tc else {
                    return Err(format!("AttributeError: messages[{i}].tool_calls must be a list of objects"));
                };
                let mut tc = tc.clone();
                match tc.get_mut("function") {
                    None => {}
                    Some(Value::Object(f)) => {
                        if let Some(Value::String(args)) = f.get("arguments") {
                            let v = py_loads(if args.is_empty() { "{}" } else { args })
                                .unwrap_or_else(|_| json!({"input": args}));
                            f.insert("arguments".into(), v);
                        }
                    }
                    Some(_) => return Err(format!("AttributeError: messages[{i}]: a tool call's function must be an object")),
                }
                parsed.push(Value::Object(tc));
            }
            m.insert("tool_calls".into(), Value::Array(parsed));
        }
        out.push(Value::Object(m));
    }
    Ok(out)
}

/// The prompt text for OpenAI-shaped `messages` and `tools` (GlmTemplate.render). `effort`
/// "low" or "high" is honored; anything else, and none, is Max.
pub fn render(messages: &[Value], tools: Option<&Value>, effort: Option<&str>) -> Result<String, String> {
    let msgs = prepare(messages)?;
    let tools = match tools {
        Some(t) if truthy(t) => Some(t.as_array().ok_or("TypeError: tools must be a list")?.as_slice()),
        _ => None,
    };
    let tpl = Tpl { msgs: &msgs, tools };
    let mut out = String::from(PREFIX);
    out.push_str(SYSTEM);
    out.push_str("Reasoning Effort: ");
    out.push_str(match effort {
        Some("low") => "Low",
        Some("high") => "High",
        _ => "Max",
    });
    if let Some(tools) = tools {
        out.push_str(TOOLS_HEAD);
        for tool in tools {
            let tool = unwrap_tool(tool)?;
            if !tool.attr("defer_loading")?.truthy() {
                out.push_str(&tool_to_json(tool)?);
                out.push('\n');
            }
        }
        out.push_str(TOOLS_TAIL);
    }
    for (i, m) in msgs.iter().enumerate() {
        let r = role(m);
        if r.is("user") {
            out.push_str(USER);
            out.push_str(&visible_text(content(m)));
        } else if r.is("assistant") {
            out.push_str(&tpl.assistant(m)?);
        } else if r.is("tool") {
            if i == 0 || !role(&msgs[i - 1]).is("tool") {
                out.push_str(&tpl.tool_block(i)?);
            }
        } else if r.is("system") {
            out.push_str(SYSTEM);
            out.push_str(&visible_text(content(m)));
        }
    }
    out.push_str(GENERATION_PROMPT);
    Ok(out)
}

// ---------------------------------------------------------------- the answer

/// The marker ids the parser routes on and the stop ids (GlmTemplate.__init__: the tokenizer's
/// ids, generation_config.json's eos_token_id).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlmIds {
    pub think: u32,
    pub think_end: u32,
    pub call: u32,
    pub call_end: u32,
    pub stop: Vec<u32>,
}

impl GlmIds {
    pub fn load(model_dir: &Path, text: &Text) -> Result<Self, Error> {
        let id = |t: &str| text.tokenizer.token_to_id(t).ok_or_else(|| format!("the tokenizer has no {t} token"));
        let gen = model_dir.join("generation_config.json");
        let gen: Value = if gen.exists() { serde_json::from_slice(&std::fs::read(gen)?)? } else { Value::Null };
        let stop = match &gen["eos_token_id"] {
            Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
            Value::Number(n) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        Ok(GlmIds { think: id(THINK)?, think_end: id(THINK_END)?, call: id(TOOL_CALL)?, call_end: id(TOOL_CALL_END)?, stop })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Reasoning,
    Content,
}

/// GlmParser: (channel, text delta) events from the answer's ids, one id at a time.
///
/// A delta is the text of the ids since the last one, held back while it ends in U+FFFD (a
/// character split across ids). GlmParser decodes the whole run each time and slices off what
/// it sent; this decodes only the ids since the last delta, which is the same text: the
/// tokenizer's decoder (ByteLevel) is lossy UTF-8 of the ids' concatenated bytes, and a
/// released text ends in a complete character, so the bytes before it never change meaning.
pub struct GlmParser {
    ids: GlmIds,
    pub chan: Channel,
    /// the current run's ids not yet released as text
    body: Vec<u32>,
    pub reasoning: String,
    pub content: String,
    /// each tool call's ids between `<tool_call>` and `</tool_call>`
    pub calls: Vec<Vec<u32>>,
    in_call: bool,
}

impl GlmParser {
    pub fn new(ids: &GlmIds) -> Self {
        GlmParser {
            ids: ids.clone(),
            chan: Channel::Reasoning,
            body: Vec::new(),
            reasoning: String::new(),
            content: String::new(),
            calls: Vec::new(),
            in_call: false,
        }
    }

    fn flush(&mut self, text: &Text, last: bool) -> Vec<(Channel, String)> {
        let delta = text.decode(&self.body).unwrap_or_default();
        if !last && delta.ends_with('\u{FFFD}') {
            return Vec::new();
        }
        self.body.clear();
        if delta.is_empty() {
            return Vec::new();
        }
        match self.chan {
            Channel::Reasoning => self.reasoning.push_str(&delta),
            Channel::Content => self.content.push_str(&delta),
        }
        vec![(self.chan, delta)]
    }

    fn switch(&mut self, text: &Text, chan: Channel) -> Vec<(Channel, String)> {
        let out = self.flush(text, true);
        self.chan = chan;
        self.body.clear();
        out
    }

    pub fn feed(&mut self, text: &Text, tok: u32) -> Vec<(Channel, String)> {
        if self.ids.stop.contains(&tok) {
            return Vec::new();
        }
        if self.in_call {
            if tok == self.ids.call_end {
                self.in_call = false;
            } else if let Some(c) = self.calls.last_mut() {
                c.push(tok);
            }
            return Vec::new();
        }
        if tok == self.ids.call {
            let out = self.switch(text, Channel::Content);
            self.in_call = true;
            self.calls.push(Vec::new());
            return out;
        }
        if tok == self.ids.think_end {
            return self.switch(text, Channel::Content);
        }
        if tok == self.ids.think {
            return self.switch(text, Channel::Reasoning);
        }
        self.body.push(tok);
        self.flush(text, false)
    }

    /// The answer is over: what was held back, a split character too.
    pub fn finish(&mut self, text: &Text) -> Vec<(Channel, String)> {
        self.flush(text, true)
    }

    /// Each tool call's decoded text (GlmParser.messages' "tool" entries)
    pub fn call_texts(&self, text: &Text) -> Vec<String> {
        self.calls.iter().map(|c| text.decode(c).unwrap_or_default()).collect()
    }

    /// (name, arguments JSON) of each tool call (GlmParser.tool_calls without the ids)
    pub fn tool_calls(&self, text: &Text, tools: Option<&Value>) -> Vec<(String, String)> {
        let types = ToolTypes::new(tools);
        self.call_texts(text).iter().map(|t| parse_tool_call(t, &types)).collect()
    }
}

/// Each tool's parameter -> its schema "type", by tool name (a later same-named tool wins)
pub struct ToolTypes(HashMap<String, HashMap<String, Value>>);

impl ToolTypes {
    pub fn new(tools: Option<&Value>) -> Self {
        let mut types = HashMap::new();
        for tl in tools.and_then(|t| t.as_array()).into_iter().flatten() {
            let f = tl.get("function").unwrap_or(tl);
            let props = f.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.as_object());
            let Some(name) = f.get("name").and_then(|n| n.as_str()) else { continue };
            let t = props
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.clone(), v.get("type").cloned().unwrap_or(Value::Null)))
                .collect();
            types.insert(name.to_string(), t);
        }
        ToolTypes(types)
    }

    fn is_string(&self, tool: &str, arg: &str) -> bool {
        self.0.get(tool).and_then(|t| t.get(arg)).and_then(|t| t.as_str()) == Some("string")
    }
}

/// `re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", text, re.S)`
fn arg_pairs(text: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(s) = text[pos..].find(ARG_KEY).map(|i| pos + i) {
        let key = s + ARG_KEY.len();
        let mut hit = None;
        let mut from = key;
        // the lazy key ends at the first </arg_key> that whitespace and <arg_value> follow
        while let Some(e) = text[from..].find(ARG_KEY_END).map(|i| from + i) {
            let after = e + ARG_KEY_END.len();
            let v = after + text[after..].find(|c: char| !py_isspace(c)).unwrap_or(text.len() - after);
            if text[v..].starts_with(ARG_VALUE) {
                let val = v + ARG_VALUE.len();
                // no </arg_value> after this value: none after a later one either
                hit = text[val..].find(ARG_VALUE_END).map(|j| (e, val, val + j));
                break;
            }
            from = e + 1;
        }
        match hit {
            Some((e, val, end)) => {
                out.push((&text[key..e], &text[val..end]));
                pos = end + ARG_VALUE_END.len();
            }
            None => pos = s + 1,
        }
    }
    out
}

/// One call's text `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`: the name (stripped)
/// and the arguments as json.dumps(ensure_ascii=False) writes them. A value whose schema type
/// is "string" stays raw; any other is JSON when it parses, else raw.
pub fn parse_tool_call(text: &str, types: &ToolTypes) -> (String, String) {
    let name = py_strip(text.split(ARG_KEY).next().unwrap_or("")).to_string();
    let mut args = Map::new();
    for (k, v) in arg_pairs(text) {
        let k = py_strip(k).to_string();
        let val = if types.is_string(&name, &k) {
            Value::String(v.to_string())
        } else {
            py_loads(v).unwrap_or_else(|_| Value::String(v.to_string()))
        };
        args.insert(k, val);
    }
    (name, py_json(&Value::Object(args)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::py_float;

    #[test]
    fn python_values() {
        assert_eq!(py_strip("\u{a0} x\u{2003}\u{1c}"), "x");
        assert_eq!(py_strip("\u{200b}x"), "\u{200b}x");
        assert_eq!(py_repr(&json!({"a": [1, 2.5, true, null], "it's": "q\"\n\u{a0}é"})),
                   r#"{'a': [1, 2.5, True, None], "it's": 'q"\n\xa0é'}"#);
        assert!(py_eq(&json!(1), &json!(1.0)) && py_eq(&json!(true), &json!(1)) && !py_eq(&json!("1"), &json!(1)));
        assert_eq!(py_float(1e16), "1e+16");
        // json.loads reads -0 as the int 0, serde_json as the float -0.0
        let v = py_loads(r#"[-0, "-0", -0.0, -0e1, {"-0": -0}, "a\"-0"]"#).unwrap();
        assert_eq!(py_json(&v), r#"[0, "-0", -0.0, -0.0, {"-0": 0}, "a\"-0"]"#);
    }

    #[test]
    fn arg_pairs_follow_the_lazy_regex() {
        let t = "f<arg_key>a</arg_key>\n <arg_value>1</arg_value><arg_key>b</arg_key>x<arg_value>2</arg_value><arg_key>c</arg_key><arg_value>3</arg_value>";
        assert_eq!(arg_pairs(t), vec![("a", "1"), ("b</arg_key>x<arg_value>2</arg_value><arg_key>c", "3")]);
        assert_eq!(arg_pairs("<arg_key>k</arg_key><arg_value>open"), vec![]);
        assert_eq!(arg_pairs("<arg_key><arg_key>k</arg_key><arg_value>v</arg_value>"), vec![("<arg_key>k", "v")]);
    }
}
