//! POST /v1/messages and /v1/messages/count_tokens: server/anthropic_api.py, so Claude Code
//! (ANTHROPIC_BASE_URL) can drive either family's engine.
//!
//! A request is translated into the OpenAI message shape (to_chat) and rendered by the
//! family's own prompt rules; the answer goes through the family's router ([`Answer`]); only
//! the envelope is Anthropic's.
//!
//! - `system` (a string or text blocks) becomes one system message; Claude Code's attribution
//!   block (`x-anthropic-billing-header: ...`) is dropped - a per-request fingerprint at the
//!   top of the prompt would cost the prefix cache the whole conversation.
//! - `tool_use` blocks become `tool_calls`, `tool_result` blocks `tool` messages ahead of the
//!   same user turn's text; images and documents a one-line placeholder (images pass as
//!   `image_url` parts on a DeepSeek server with a vision tower).
//! - Reasoning is never returned. GLM's template re-renders every earlier turn's `<think>`, so
//!   a client that drops it would make each prompt diverge from the cached one at the previous
//!   answer; [`ReasoningMemo`] keeps each answer's reasoning by its tool-use ids (else its
//!   text) and puts it back when the client sends the turn again.
//! - `thinking`, `output_config`, `cache_control`, `tool_choice` and server tools are accepted
//!   and ignored; the effort is the server's.
//! - usage: `cache_read_input_tokens` is the engine's reused prefix, `input_tokens` the rest.
//!
//! A stream sends message_start at once, a ping every 5 s while nothing else goes out
//! (prefill, reasoning, tool bodies), the text block from the first non-whitespace text, the
//! tool_use blocks at the end, then message_delta and message_stop.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tidelm_engine::decode::{Event, Finish};
use tidelm_text::chat::{py_json, py_loads};
use tidelm_text::glm::{py_isspace, py_strip, truthy};

use crate::api::{body_json, submit, Failure, Request};
use crate::http::{send_json, start_events, ApiError};
use crate::output::{hex24, Answer, Phase};
use crate::{Cmd, Ctx, Family};

pub const ATTRIBUTION: &str = "x-anthropic-billing-header";
/// an SSE ping after this long without an event
pub const PING: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------- the reasoning memo

#[derive(Clone, Debug, PartialEq, Eq)]
enum Key {
    /// a tool-use id, as JSON (an id of another type is another key, as in Python)
    Id(String),
    /// a digest of the answer's stripped text (Python's is sha1; any digest keys the same)
    Text([u8; 32]),
}

/// A bounded map (least recently used out) from an answer's tool-use ids and text to the
/// reasoning that produced it.
pub struct ReasoningMemo {
    size: usize,
    /// oldest first
    map: Mutex<Vec<(Key, Arc<str>)>>,
}

impl Default for ReasoningMemo {
    fn default() -> Self {
        ReasoningMemo::new(512)
    }
}

impl ReasoningMemo {
    pub fn new(size: usize) -> Self {
        ReasoningMemo { size, map: Mutex::new(Vec::new()) }
    }

    fn keys(text: &str, ids: &[Value]) -> Vec<Key> {
        let mut keys: Vec<Key> = ids.iter().filter(|i| truthy(i)).map(|i| Key::Id(i.to_string())).collect();
        let t = py_strip(text);
        if !t.is_empty() {
            keys.push(Key::Text(Sha256::digest(t.as_bytes()).into()));
        }
        keys
    }

    pub fn put(&self, text: &str, ids: &[Value], reasoning: &str) {
        if reasoning.is_empty() {
            return;
        }
        let r: Arc<str> = reasoning.into();
        let mut map = self.map.lock().unwrap();
        for k in Self::keys(text, ids) {
            map.retain(|(x, _)| *x != k);
            map.push((k, r.clone()));
        }
        let over = map.len().saturating_sub(self.size);
        map.drain(..over);
    }

    pub fn get(&self, text: &str, ids: &[Value]) -> Option<String> {
        let mut map = self.map.lock().unwrap();
        for k in Self::keys(text, ids) {
            if let Some(i) = map.iter().position(|(x, _)| *x == k) {
                let e = map.remove(i);
                let r = e.1.to_string();
                map.push(e);
                return Some(r);
            }
        }
        None
    }
}

// ---------------------------------------------------------------- the request

fn type_is(b: &Map<String, Value>, t: &str) -> bool {
    b.get("type").and_then(|v| v.as_str()) == Some(t)
}

/// `_blocks`: a string is one text block; a list's objects; nothing else holds blocks
fn blocks(content: Option<&Value>) -> Result<Vec<Map<String, Value>>, String> {
    match content {
        Some(Value::String(s)) => Ok(vec![json!({"type": "text", "text": s}).as_object().unwrap().clone()]),
        Some(Value::Array(a)) => Ok(a.iter().filter_map(|b| b.as_object().cloned()).collect()),
        // a dict iterates its keys, none of them a block
        None | Some(Value::Object(_)) => Ok(Vec::new()),
        Some(v) if !truthy(v) => Ok(Vec::new()),
        Some(v) => Err(format!("TypeError: content {v} is not a list of blocks")),
    }
}

/// a text block's `text` (missing: ""); `strict`: a falsy non-string is an error too (a join
/// of it fails), else it is dropped (None)
fn block_text(b: &Map<String, Value>, strict: bool) -> Result<Option<String>, String> {
    match b.get("text") {
        None => Ok(Some(String::new())),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) if !strict && !truthy(v) => Ok(None),
        Some(v) => Err(format!("TypeError: a text block's text must be a string, not {v}")),
    }
}

/// `_text`: text blocks and a placeholder for each image and document, "\n\n"-joined
fn text_of(content: Option<&Value>) -> Result<String, String> {
    let mut out = Vec::new();
    for b in blocks(content)? {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => out.extend(block_text(&b, false)?),
            Some(kind @ ("image" | "document")) => out.push(format!("[{kind} omitted: this model reads text only]")),
            _ => {}
        }
    }
    Ok(out.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n"))
}

/// an f-string's `{v}` of a JSON value (str() of it; containers as JSON, near enough here)
fn fstr(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => py_json(other),
    }
}

/// `_image_url`: a base64 source's data URL, or a url source's URL
fn image_url(b: &Map<String, Value>) -> Option<String> {
    let src = b.get("source").filter(|s| truthy(s))?;
    let get = |k: &str| src.get(k).filter(|v| truthy(v));
    match src.get("type").and_then(|t| t.as_str()) {
        Some("base64") => get("data").map(|d| {
            format!("data:{};base64,{}", src.get("media_type").map(fstr).unwrap_or_else(|| "image/png".into()), fstr(d))
        }),
        Some("url") => get("url").map(fstr),
        _ => None,
    }
}

/// `_parts`: a user turn's content as OpenAI parts when it shows an image, else its text
fn parts(blocks: &[Map<String, Value>]) -> Result<Value, String> {
    let as_value = |b: &Map<String, Value>| Value::Array(vec![Value::Object(b.clone())]);
    if !blocks.iter().any(|b| type_is(b, "image") && image_url(b).is_some()) {
        return Ok(Value::String(text_of(Some(&Value::Array(blocks.iter().cloned().map(Value::Object).collect())))?));
    }
    let mut out = Vec::new();
    for b in blocks {
        let url = if type_is(b, "image") { image_url(b) } else { None };
        if let Some(url) = url {
            out.push(json!({"type": "image_url", "image_url": {"url": url}}));
        } else {
            let t = text_of(Some(&as_value(b)))?;
            if !t.is_empty() {
                out.push(json!({"type": "text", "text": t}));
            }
        }
    }
    Ok(Value::Array(out))
}

/// to_chat: (OpenAI-style messages, OpenAI-style tools) for an Anthropic Messages request
pub fn to_chat(req: &Value, memo: Option<&ReasoningMemo>, images: bool) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut msgs = Vec::new();
    let system = match req.get("system") {
        Some(Value::Array(_)) => {
            let mut texts = Vec::new();
            for b in blocks(req.get("system"))? {
                if type_is(&b, "text") {
                    let t = block_text(&b, true)?.unwrap_or_default();
                    if !t.starts_with(ATTRIBUTION) {
                        texts.push(t);
                    }
                }
            }
            Value::String(texts.join("\n\n"))
        }
        Some(v) => v.clone(),
        None => Value::Null,
    };
    if truthy(&system) {
        msgs.push(json!({"role": "system", "content": system}));
    }
    let messages: &[Value] = match req.get("messages") {
        Some(Value::Array(a)) => a,
        Some(v) if truthy(v) => return Err("TypeError: messages must be a list".into()),
        _ => &[],
    };
    for m in messages {
        let Some(m) = m.as_object() else {
            return Err("AttributeError: each message must be an object".into());
        };
        let role = m.get("role");
        let blocks = blocks(m.get("content"))?;
        if role.and_then(|r| r.as_str()) == Some("assistant") {
            let mut text = String::new();
            for b in blocks.iter().filter(|b| type_is(b, "text")) {
                text.push_str(&block_text(b, true)?.unwrap_or_default());
            }
            let uses: Vec<&Map<String, Value>> = blocks.iter().filter(|b| type_is(b, "tool_use")).collect();
            let mut msg = json!({"role": "assistant", "content": text});
            if !uses.is_empty() {
                msg["tool_calls"] = uses
                    .iter()
                    .map(|u| {
                        let input = u.get("input").filter(|i| truthy(i)).cloned().unwrap_or_else(|| json!({}));
                        json!({"id": u.get("id").cloned().unwrap_or(Value::Null), "type": "function",
                               "function": {"name": u.get("name").cloned().unwrap_or(Value::Null), "arguments": py_json(&input)}})
                    })
                    .collect();
            }
            let ids: Vec<Value> = uses.iter().map(|u| u.get("id").cloned().unwrap_or(Value::Null)).collect();
            if let Some(r) = memo.and_then(|memo| memo.get(&text, &ids)) {
                msg["reasoning_content"] = Value::String(r);
            }
            msgs.push(msg);
            continue;
        }
        let results: Vec<&Map<String, Value>> = blocks.iter().filter(|b| type_is(b, "tool_result")).collect();
        for b in &results {
            msgs.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                             "content": text_of(b.get("content"))?}));
        }
        let others: Vec<Map<String, Value>> = blocks.iter().filter(|b| !type_is(b, "tool_result")).cloned().collect();
        let system = role.and_then(|r| r.as_str()) == Some("system");
        let rest = if images && !system {
            parts(&others)?
        } else {
            Value::String(text_of(Some(&Value::Array(others.into_iter().map(Value::Object).collect())))?)
        };
        if truthy(&rest) || results.is_empty() {
            msgs.push(json!({"role": if system { "system" } else { "user" }, "content": rest}));
        }
    }
    let tools = match req.get("tools") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|t| t.as_object())
            .filter(|t| t.contains_key("input_schema"))
            .map(|t| json!({"type": "function", "function": {
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": t.get("description").cloned().unwrap_or_else(|| json!("")),
                "parameters": t["input_schema"].clone()}}))
            .collect(),
        Some(v @ (Value::Bool(_) | Value::Number(_))) if truthy(v) => return Err("TypeError: tools must be a list".into()),
        _ => Vec::new(),
    };
    Ok((msgs, tools))
}

// ---------------------------------------------------------------- the reply

/// OpenAI tool_calls -> Anthropic tool_use blocks
pub fn tool_uses(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| {
            let f = c.get("function").filter(|f| truthy(f)).cloned().unwrap_or_else(|| json!({}));
            let raw = f.get("arguments").filter(|a| truthy(a));
            let args = match raw {
                None => json!({}),
                Some(Value::String(s)) => py_loads(s).unwrap_or_else(|_| json!({"input": s})),
                Some(other) => json!({"input": other}),
            };
            let id = match c.get("id") {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            json!({"type": "tool_use", "id": format!("toolu_{}", id.strip_prefix("call_").unwrap_or(&id)),
                   "name": f.get("name").cloned().unwrap_or(Value::Null),
                   "input": if args.is_object() { args } else { json!({"input": args}) }})
        })
        .collect()
}

/// tool_use | max_tokens (the answer hit its token limit) | end_turn
pub fn stop_reason(uses: &[Value], length: bool) -> &'static str {
    if !uses.is_empty() {
        "tool_use"
    } else if length {
        "max_tokens"
    } else {
        "end_turn"
    }
}

pub fn usage_counts(prompt_tokens: usize, reused: usize, n: usize) -> Value {
    json!({"input_tokens": prompt_tokens.saturating_sub(reused), "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": reused, "output_tokens": n})
}

/// A whole (non-streamed) reply.
pub fn message(mid: &str, model: &str, text: &str, uses: Vec<Value>, stop: &str, usage: Value) -> Value {
    let mut content: Vec<Value> = if text.is_empty() { Vec::new() } else { vec![json!({"type": "text", "text": text})] };
    content.extend(uses);
    json!({"id": mid, "type": "message", "role": "assistant", "model": model, "content": content,
           "stop_reason": stop, "stop_sequence": null, "usage": usage})
}

/// Anthropic's wording: Claude Code compacts the conversation when it sees it.
pub fn too_long(prompt_tokens: usize, context: usize) -> String {
    format!("prompt is too long: {prompt_tokens} tokens > {context} maximum")
}

pub fn error_kind(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    }
}

pub fn error(message: &str, kind: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

/// One streamed reply's SSE events; constructing it sends message_start.
pub struct Stream<W: Write> {
    w: W,
    text_open: bool,
    pending: String,
    last: Instant,
}

impl<W: Write> Stream<W> {
    pub fn new(w: W, mid: &str, model: &str, prompt_tokens: usize) -> std::io::Result<Self> {
        let mut s = Stream { w, text_open: false, pending: String::new(), last: Instant::now() };
        s.event("message_start", json!({"message": {
            "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}}}))?;
        Ok(s)
    }

    /// `event: KIND` and `{"type": KIND, ...body}` as json.dumps(ensure_ascii=False) writes it
    fn event(&mut self, kind: &str, body: Value) -> std::io::Result<()> {
        let mut m = Map::new();
        m.insert("type".into(), json!(kind));
        if let Value::Object(b) = body {
            m.extend(b);
        }
        write!(self.w, "event: {kind}\ndata: {}\n\n", py_json(&Value::Object(m)))?;
        self.w.flush()?;
        self.last = Instant::now();
        Ok(())
    }

    /// A ping now, or (`idle`) only when nothing went out for PING: reasoning, tool-call bodies
    /// and prefill show the client nothing, and its idle watchdog would give up.
    pub fn ping(&mut self, idle: bool) -> std::io::Result<()> {
        if !idle || self.last.elapsed() > PING {
            self.event("ping", json!({}))?;
        }
        Ok(())
    }

    /// Visible text. No block until there is something to show: the newline a model writes
    /// after its reasoning, ahead of a tool call, would otherwise be an empty text block.
    pub fn text(&mut self, delta: &str) -> std::io::Result<()> {
        let mut delta = delta.to_string();
        if !self.text_open {
            self.pending.push_str(&delta);
            if py_strip(&self.pending).is_empty() {
                return Ok(());
            }
            delta = self.pending.trim_start_matches(py_isspace).to_string();
            self.pending.clear();
            self.event("content_block_start", json!({"index": 0, "content_block": {"type": "text", "text": ""}}))?;
            self.text_open = true;
        }
        self.event("content_block_delta", json!({"index": 0, "delta": {"type": "text_delta", "text": delta}}))
    }

    pub fn finish(&mut self, uses: &[Value], stop: &str, usage: Value) -> std::io::Result<()> {
        let mut index = 0;
        if self.text_open {
            self.event("content_block_stop", json!({"index": 0}))?;
            index = 1;
        }
        for (i, u) in uses.iter().enumerate() {
            let i = i + index;
            let mut block = u.clone();
            block["input"] = json!({});
            self.event("content_block_start", json!({"index": i, "content_block": block}))?;
            self.event("content_block_delta", json!({"index": i, "delta": {"type": "input_json_delta", "partial_json": py_json(&u["input"])}}))?;
            self.event("content_block_stop", json!({"index": i}))?;
        }
        self.event("message_delta", json!({"delta": {"stop_reason": stop, "stop_sequence": null}, "usage": usage}))?;
        self.event("message_stop", json!({}))
    }

    pub fn error(&mut self, message: &str) -> std::io::Result<()> {
        self.event("error", json!({"error": {"type": "api_error", "message": message}}))
    }
}

// ---------------------------------------------------------------- the route

/// How much of the context an answer may have.
pub enum Room {
    /// server/muse_app.py: max_seq - prompt - 1 (at least 1); a prompt of max_seq is refused
    Glm,
    /// server/app.py: max_seq - prompt - 8, the drafter's block of headroom; refused below 1
    Ds,
}

impl Room {
    /// The answer's max_tokens, or None: the prompt does not fit.
    pub fn fit(&self, prompt: usize, want: usize, max_seq: usize) -> Option<usize> {
        match self {
            Room::Glm if prompt >= max_seq => None,
            Room::Glm if prompt + want > max_seq => Some(max_seq.saturating_sub(prompt + 1).max(1)),
            Room::Glm => Some(want),
            Room::Ds => Some(want.min(max_seq.saturating_sub(prompt + 8))).filter(|&n| n >= 1),
        }
    }
}

/// A translated request, rendered and ready for the engine.
pub struct Prepared {
    pub request: Request,
    pub answer: Box<dyn Answer>,
    pub room: Room,
    /// the tools for tool-call argument types (GLM)
    pub tools: Value,
}

pub enum Prep {
    /// count_tokens: the prompt's length
    Count(usize),
    Run(Prepared),
}

/// The route: errors in Anthropic's shape.
pub fn serve(stream: &mut TcpStream, path: &str, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    match handle(stream, path, body, ctx) {
        Err(Failure::Api(e)) => {
            send_json(stream, e.status, &error(&e.message, error_kind(e.status)))?;
            Ok(())
        }
        r => r,
    }
}

fn handle(stream: &mut TcpStream, path: &str, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    let count = match path {
        "/v1/messages" => false,
        "/v1/messages/count_tokens" => true,
        _ => return Err(ApiError { status: 404, message: format!("no route for POST {path}"), param: None }.into()),
    };
    let req = body_json(body)?;
    if !req.is_object() {
        return Err(ApiError::bad("the request must be a JSON object", None).into());
    }
    let images = matches!(ctx.family, Family::Ds) && ctx.vision.is_some();
    let (msgs, tools) = to_chat(&req, Some(&ctx.memo), images).map_err(|e| ApiError::bad(e, Some("messages")))?;
    let prep = match &ctx.family {
        Family::Glm(g) => crate::glm::anthropic_prepare(&req, &msgs, tools, ctx, g, count)?,
        Family::Ds => crate::api::anthropic_prepare(&req, msgs, tools, ctx, count)?,
    };
    let mut p = match prep {
        Prep::Count(n) => {
            send_json(stream, 200, &json!({"input_tokens": n}))?;
            return Ok(());
        }
        Prep::Run(p) => p,
    };
    let n_prompt = p.request.prompt.len();
    p.request.max_tokens = p.room.fit(n_prompt, p.request.max_tokens, ctx.max_seq)
        .ok_or_else(|| ApiError::bad(too_long(n_prompt, ctx.max_seq), Some("messages")))?;
    let streaming = req.get("stream").is_some_and(truthy);
    let mid = format!("msg_{}", hex24());
    let tools = p.tools.clone();
    let tools = Some(&tools).filter(|t| !t.is_null());
    let mut answer = p.answer;
    let (id, rx) = submit(ctx, p.request)?;
    if !streaming {
        let (finish, gen) = loop {
            match rx.recv() {
                Ok(Event::Tokens(t)) => {
                    answer.feed(&ctx.text, &t);
                }
                Ok(Event::Done(Finish::Error(e), _)) => return Err(ApiError { status: 500, message: e, param: None }.into()),
                Ok(Event::Done(f, g)) => break (f, g),
                Err(_) => return Err(ApiError { status: 500, message: "the engine dropped the request".into(), param: None }.into()),
            }
        };
        answer.finish(&ctx.text);
        let (calls, _) = answer.tool_calls(&ctx.text, tools);
        let text = py_strip(answer.content()).to_string();
        let uses = tool_uses(&calls);
        let ids: Vec<Value> = uses.iter().map(|u| u["id"].clone()).collect();
        ctx.memo.put(&text, &ids, answer.reasoning());
        let length = !answer.ended() && matches!(finish, Finish::Length);
        let stop = stop_reason(&uses, length);
        let usage = usage_counts(n_prompt, gen.cached, answer.tokens());
        send_json(stream, 200, &message(&mid, &ctx.model, &text, uses, stop, usage))?;
        return Ok(());
    }
    start_events(stream)?;
    let gone = |_: std::io::Error| {
        let _ = ctx.engine.send(Cmd::Cancel(id));
        Failure::Io
    };
    let mut out = Stream::new(&mut *stream, &mid, &ctx.model, n_prompt).map_err(gone)?;
    let (finish, gen) = loop {
        match rx.recv_timeout(PING) {
            Err(RecvTimeoutError::Timeout) => out.ping(false).map_err(gone)?,
            Err(RecvTimeoutError::Disconnected) => {
                let _ = out.error("the engine dropped the request");
                return Ok(());
            }
            Ok(Event::Tokens(t)) => {
                for (ph, d) in answer.feed(&ctx.text, &t) {
                    if ph == Phase::Content {
                        out.text(&d).map_err(gone)?;
                    }
                }
                out.ping(true).map_err(gone)?;
            }
            Ok(Event::Done(Finish::Error(e), _)) => {
                let _ = out.error(&e);
                return Ok(());
            }
            Ok(Event::Done(f, g)) => break (f, g),
        }
    };
    for (ph, d) in answer.finish(&ctx.text) {
        if ph == Phase::Content {
            out.text(&d).map_err(gone)?;
        }
    }
    let (calls, unparsed) = answer.tool_calls(&ctx.text, tools);
    // a block no parser could read is content the stream has not sent
    if !unparsed.is_empty() {
        out.text(&unparsed).map_err(gone)?;
    }
    let text = py_strip(answer.content()).to_string();
    let uses = tool_uses(&calls);
    let ids: Vec<Value> = uses.iter().map(|u| u["id"].clone()).collect();
    ctx.memo.put(&text, &ids, answer.reasoning());
    let length = !answer.ended() && matches!(finish, Finish::Length);
    out.finish(&uses, stop_reason(&uses, length), usage_counts(n_prompt, gen.cached, answer.tokens())).map_err(gone)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// server/test_anthropic_api.py's REQ
    pub fn req() -> Value {
        json!({
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=abc;"},
                       {"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "Bash", "description": "Run a command",
                       "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
                      {"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": "List the files."}],
            "max_tokens": 512, "thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
        })
    }

    #[test]
    fn system_tools_and_turns() {
        let mut r = req();
        r["messages"] = json!([
            {"role": "user", "content": [{"type": "text", "text": "Look."},
                                         {"type": "image", "source": {"type": "base64", "data": ""}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Running."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": [{"type": "text", "text": "a.txt"}]},
                                         {"type": "text", "text": "<system-reminder>x</system-reminder>"}]},
            {"role": "system", "content": "Mid-conversation note."}]);
        let (msgs, tools) = to_chat(&r, None, false).unwrap();
        assert_eq!(msgs[0], json!({"role": "system", "content": "You are an agent."}));
        assert_eq!(msgs[1]["content"], "Look.\n\n[image omitted: this model reads text only]");
        // json.dumps({"command": "ls"})
        assert_eq!(msgs[2]["tool_calls"][0]["function"], json!({"name": "Bash", "arguments": "{\"command\": \"ls\"}"}));
        assert!(msgs[2].get("reasoning_content").is_none());
        // the result first, then the same turn's text
        assert_eq!(msgs[3], json!({"role": "tool", "tool_call_id": "toolu_1", "content": "a.txt"}));
        assert_eq!(msgs[4], json!({"role": "user", "content": "<system-reminder>x</system-reminder>"}));
        assert_eq!(msgs[5], json!({"role": "system", "content": "Mid-conversation note."}));
        assert_eq!(tools.iter().map(|t| t["function"]["name"].clone()).collect::<Vec<_>>(), vec![json!("Bash")]);
        assert_eq!(tools[0], json!({"type": "function", "function": {"name": "Bash", "description": "Run a command",
                                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}));
    }

    #[test]
    fn images_pass_as_parts_with_a_tower() {
        let r = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is it?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
            {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}},
            {"type": "image", "source": {"type": "base64", "data": ""}},
            {"type": "document"}]}]});
        let (msgs, _) = to_chat(&r, None, true).unwrap();
        assert_eq!(msgs[0]["content"], json!([
            {"type": "text", "text": "What is it?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            {"type": "text", "text": "[image omitted: this model reads text only]"},
            {"type": "text", "text": "[document omitted: this model reads text only]"}]));
        // no tower, or a system turn: text
        let (msgs, _) = to_chat(&r, None, false).unwrap();
        assert_eq!(msgs[0]["content"], "What is it?\n\n[image omitted: this model reads text only]\n\n[image omitted: this model \
                                        reads text only]\n\n[image omitted: this model reads text only]\n\n[document omitted: \
                                        this model reads text only]");
    }

    #[test]
    fn a_turn_of_only_results_has_no_user_message() {
        let r = json!({"system": "", "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ra"},
                                         {"type": "tool_result", "tool_use_id": "b"}]},
            {"role": "user", "content": []},
            {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": null}, {"type": "text", "text": "x"}]}]});
        let (msgs, _) = to_chat(&r, None, false).unwrap();
        assert_eq!(Value::Array(msgs), json!([
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": ""},
            {"role": "user", "content": ""},
            {"role": "user", "content": "x"}]));
    }

    #[test]
    fn memo_restores_reasoning_by_id_or_text() {
        let memo = ReasoningMemo::new(4);
        memo.put("Running.", &[json!("toolu_1")], "I should run ls.");
        memo.put("  Done.\n", &[], "All finished.");
        assert_eq!(memo.get("", &[json!("toolu_1")]).as_deref(), Some("I should run ls."));
        assert_eq!(memo.get("Done.", &[]).as_deref(), Some("All finished."));
        assert_eq!(memo.get("Other.", &[json!("toolu_2")]), None);
        for i in 0..8 {
            memo.put(&format!("t{i}"), &[], "r");
        }
        assert_eq!(memo.get("", &[json!("toolu_1")]), None); // evicted
        // a read refreshes: of three keys, the one read survives the next put
        let memo = ReasoningMemo::new(2);
        memo.put("a", &[], "ra");
        memo.put("b", &[], "rb");
        assert!(memo.get("a", &[]).is_some());
        memo.put("c", &[], "rc");
        assert_eq!((memo.get("a", &[]).is_some(), memo.get("b", &[]).is_some()), (true, false));
        // no reasoning, no entry
        memo.put("d", &[], "");
        assert_eq!(memo.get("d", &[]), None);
    }

    #[test]
    fn tool_uses_and_stop_reasons() {
        let uses = tool_uses(&[json!({"id": "call_abc", "type": "function", "function": {"name": "Bash", "arguments": "{\"command\": \"ls\"}"}}),
                               json!({"id": "x", "function": {"name": "f", "arguments": "not json"}}),
                               json!({"id": "call_y", "function": {"name": "g", "arguments": "[1]"}})]);
        assert_eq!(uses[0], json!({"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {"command": "ls"}}));
        assert_eq!(uses[1]["input"], json!({"input": "not json"}));
        assert_eq!((uses[1]["id"].as_str(), &uses[2]["input"]), (Some("toolu_x"), &json!({"input": [1]})));
        assert_eq!((stop_reason(&uses, true), stop_reason(&[], true), stop_reason(&[], false)), ("tool_use", "max_tokens", "end_turn"));
        assert_eq!(usage_counts(100, 7, 5), json!({"input_tokens": 93, "cache_creation_input_tokens": 0,
                                                     "cache_read_input_tokens": 7, "output_tokens": 5}));
        assert_eq!((Room::Glm.fit(10, 100, 50), Room::Glm.fit(49, 5, 50), Room::Glm.fit(50, 5, 50)), (Some(39), Some(1), None));
        assert_eq!((Room::Ds.fit(10, 100, 50), Room::Ds.fit(41, 5, 50), Room::Ds.fit(42, 5, 50)), (Some(32), Some(1), None));
    }

    #[test]
    fn stream_events_in_order() {
        let mut buf = Vec::new();
        {
            let mut s = Stream::new(&mut buf, "msg_1", "glm", 12).unwrap();
            s.text("\n").unwrap(); // no block for whitespace
            s.text("  Let me").unwrap();
            s.text(" look.").unwrap();
            let uses = vec![json!({"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls -la"}})];
            s.finish(&uses, "tool_use", usage_counts(12, 7, 3)).unwrap();
        }
        let text = String::from_utf8(buf).unwrap();
        let events: Vec<Value> = text.trim_end().split("\n\n").map(|chunk| {
            let (kind, data) = chunk.split_once('\n').unwrap();
            let v: Value = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
            assert_eq!(kind, format!("event: {}", v["type"].as_str().unwrap()));
            v
        }).collect();
        let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                           "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
        assert_eq!((events[2]["delta"]["text"].as_str(), events[3]["delta"]["text"].as_str()), (Some("Let me"), Some(" look.")));
        assert_eq!(events[5], json!({"type": "content_block_start", "index": 1,
                                     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}}));
        assert_eq!(events[6]["delta"], json!({"type": "input_json_delta", "partial_json": "{\"command\": \"ls -la\"}"}));
        assert_eq!(events[8]["usage"]["cache_read_input_tokens"], 7);
        // the bytes json.dumps writes: ", " and ": " separators, type first
        assert!(text.starts_with("event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_1\", "));
    }
}
