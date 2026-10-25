//! POST /v1/chat/completions and /v1/debug/prompt: server/app.py's request rules
//! (parse_sampling, resolve_thinking, build_chat_prompt) over the Rust port of the checkpoint's
//! encoder, the answer routed into reasoning, content and tool calls (output.rs) and streamed
//! as OpenAI chunks with usage.

use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tidelm_engine::decode::{Event, Finish, Job};
use tidelm_engine::forward::Sampling;
use tidelm_text::chat::{encode_messages, py_json, IMAGE};
use tidelm_text::Text;

use crate::http::{send_event, send_json, start_events, ApiError};
use crate::output::{OutputRouter, Phase};
use crate::{Cmd, Ctx};

/// A request that failed: with an answer for the client, or on the connection itself.
pub enum Failure {
    Api(ApiError),
    Io,
}

impl From<std::io::Error> for Failure {
    fn from(_: std::io::Error) -> Self {
        Failure::Io
    }
}

impl From<ApiError> for Failure {
    fn from(e: ApiError) -> Self {
        Failure::Api(e)
    }
}

/// server/app.py's defaults
const DEFAULT_MAX_TOKENS: usize = 131072;
const DEFAULT_TEMPERATURE: f64 = 1.0;
const DEFAULT_TOP_P: f64 = 0.95;
const EFFORT_ALIASES: [(&str, u32); 5] = [("low", 50), ("medium", 60), ("high", 75), ("xhigh", 90), ("max", 100)];
const NO_THINKING_EFFORTS: [&str; 2] = ["none", "low"];

struct ChatReq {
    /// as the encoder takes them: developer folded into system, typed image tokens escaped,
    /// the tools and a response schema on the first (system) message
    messages: Vec<Value>,
    thinking: bool,
    effort: u32,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    stop: Vec<String>,
    seed: Option<u64>,
    ignore_eos: bool,
    stream: bool,
    include_usage: bool,
}

fn num(body: &Value, key: &'static str, default: f64, lo: f64, hi: f64) -> Result<f64, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN);
            if (lo..=hi).contains(&v) {
                Ok(v)
            } else {
                Err(ApiError::bad(format!("`{key}` must be in [{lo}, {hi}]"), Some(key)))
            }
        }
        _ => Err(ApiError::bad(format!("`{key}` must be a number"), Some(key))),
    }
}

/// resolve_thinking: chat_template_kwargs.thinking, then enable_thinking, then
/// chat_template_kwargs.enable_thinking; the effort from reasoning.effort, reasoning_effort or
/// chat_template_kwargs.reasoning_effort (a name or 1..=100, which also sets thinking unless
/// thinking was given); the server's defaults otherwise.
fn resolve_thinking(body: &Value, default_thinking: bool, default_effort: u32) -> Result<(bool, u32), ApiError> {
    let ctk = body.get("chat_template_kwargs").cloned().unwrap_or(Value::Null);
    if !ctk.is_null() && !ctk.is_object() {
        return Err(ApiError::bad("`chat_template_kwargs` must be an object", Some("chat_template_kwargs")));
    }
    let mut thinking = None;
    for (src, key) in [(&ctk, "thinking"), (body, "enable_thinking"), (&ctk, "enable_thinking")] {
        match src.get(key) {
            None | Some(Value::Null) => continue,
            Some(Value::Bool(b)) => {
                thinking = Some(*b);
                break;
            }
            Some(_) => return Err(ApiError::bad(format!("`{key}` must be a boolean"), None)),
        }
    }
    let raw = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .filter(|v| !v.is_null())
        .or_else(|| body.get("reasoning_effort").filter(|v| !v.is_null()))
        .or_else(|| ctk.get("reasoning_effort").filter(|v| !v.is_null()));
    let (mut effort, mut effort_thinking) = (None, None);
    if let Some(raw) = raw {
        let as_int = match raw {
            Value::Number(n) => n.as_i64(),
            Value::String(s) if s.trim().trim_start_matches('-').chars().all(|c| c.is_ascii_digit()) && !s.trim().is_empty() => {
                s.trim().parse::<i64>().ok()
            }
            _ => None,
        };
        if let Some(e) = as_int {
            if !(1..=100).contains(&e) {
                return Err(ApiError::bad("integer reasoning effort must be within 1-100", Some("reasoning_effort")));
            }
            (effort, effort_thinking) = (Some(e as u32), Some(true));
        } else if let Value::String(s) = raw {
            let name = s.trim().to_lowercase();
            if name == "none" {
                effort_thinking = Some(false);
            } else if let Some(&(_, e)) = EFFORT_ALIASES.iter().find(|(n, _)| *n == name) {
                (effort, effort_thinking) = (Some(e), Some(!NO_THINKING_EFFORTS.contains(&name.as_str())));
            } else {
                return Err(ApiError::bad(format!("unknown reasoning effort {s:?}; use none/low/medium/high/xhigh/max or 1-100"),
                                         Some("reasoning_effort")));
            }
        } else {
            return Err(ApiError::bad("reasoning effort must be a string or an integer 1-100", Some("reasoning_effort")));
        }
    }
    Ok((thinking.or(effort_thinking).unwrap_or(default_thinking), effort.unwrap_or(default_effort)))
}

/// A typed `<｜deepseek_image｜>` with a zero-width space after the "<": the same characters
/// as ordinary text, never the image id (server/app.py's IMAGE_TOKEN_ESCAPED).
const IMAGE_ESCAPED: &str = "<\u{200b}｜deepseek_image｜>";
const NO_VISION: &str = "image inputs are not supported by this server (no vision tower loaded)";

/// escape_image_tokens: every string of a request structure; a tool call's JSON arguments are
/// decoded, escaped and written back, since a client's json.dumps may have written the bars
/// as \uff5c, which a substring replace would miss.
fn escape_images(v: &mut Value) {
    match v {
        Value::String(s) if s.contains(IMAGE) => *s = s.replace(IMAGE, IMAGE_ESCAPED),
        Value::Array(a) => a.iter_mut().for_each(escape_images),
        Value::Object(m) => {
            if let Some(Value::String(args)) = m.get_mut("function").and_then(|f| f.get_mut("arguments")) {
                *args = match serde_json::from_str::<Value>(args) {
                    Ok(mut d) => {
                        escape_images(&mut d);
                        py_json(&d)
                    }
                    Err(_) => args.replace(IMAGE, IMAGE_ESCAPED),
                };
            }
            m.values_mut().for_each(escape_images);
        }
        _ => {}
    }
}

fn is_image_block(b: &Value) -> bool {
    matches!(b.get("type").and_then(|t| t.as_str()), Some("image") | Some("image_url") | Some("input_image"))
}

fn nonempty(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        _ => true,
    }
}

/// build_chat_prompt's message rules, up to the encoder.
fn parse_messages(body: &Value, vision: bool) -> Result<Vec<Value>, ApiError> {
    let msgs = body.get("messages").and_then(|m| m.as_array()).filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("`messages` must be a non-empty list", Some("messages")))?;
    let mut msgs = msgs.clone();
    for (i, m) in msgs.iter_mut().enumerate() {
        let Some(role) = m.get("role").and_then(|r| r.as_str()) else {
            return Err(ApiError::bad(format!("messages[{i}] must be an object with a `role`"), Some("messages")));
        };
        // V4.1 has no developer role: fold it into system, mid-conversation too
        if role == "developer" {
            m["role"] = Value::String("system".into());
        }
        escape_images(m);
        if !vision && m.get("content").and_then(|c| c.as_array()).is_some_and(|c| c.iter().any(is_image_block)) {
            return Err(ApiError::bad(NO_VISION, Some("messages")));
        }
    }
    // tools and a response schema ride on the first message, which the encoder renders them
    // for only as a system message
    let mut tools = body.get("tools").filter(|t| !t.is_null()).cloned();
    if body.get("tool_choice").and_then(|c| c.as_str()) == Some("none") {
        tools = None;
    }
    if tools.as_ref().is_some_and(|t| !t.is_array()) {
        return Err(ApiError::bad("`tools` must be a list", Some("tools")));
    }
    let mut schema = None;
    if let Some(rf) = body.get("response_format").and_then(|r| r.as_object()) {
        if rf.get("type").and_then(|t| t.as_str()) == Some("json_schema") {
            schema = rf.get("json_schema").and_then(|j| j.get("schema")).filter(|s| !s.is_null()).cloned();
        }
    }
    for v in tools.iter_mut().chain(schema.iter_mut()) {
        escape_images(v);
    }
    let (tools, schema) = (tools.filter(nonempty), schema.filter(nonempty));
    if tools.is_some() || schema.is_some() {
        if msgs[0].get("role").and_then(|r| r.as_str()) != Some("system") {
            msgs.insert(0, json!({"role": "system", "content": ""}));
        }
        if let Some(t) = tools {
            msgs[0]["tools"] = t;
        }
        if let Some(s) = schema {
            msgs[0]["response_format"] = s;
        }
    }
    Ok(msgs)
}

fn parse(body: &[u8], ctx: &Ctx) -> Result<ChatReq, ApiError> {
    let body: Value = serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid JSON: {e}"), None))?;
    let messages = parse_messages(&body, ctx.vision.is_some())?;
    let (thinking, effort) = resolve_thinking(&body, ctx.default_thinking, ctx.default_effort)?;
    let mt = body.get("max_completion_tokens").filter(|v| !v.is_null()).or_else(|| body.get("max_tokens").filter(|v| !v.is_null()));
    let max_tokens = match mt {
        None => DEFAULT_MAX_TOKENS,
        Some(v) => v.as_u64().filter(|&n| n >= 1).ok_or_else(|| ApiError::bad("`max_tokens` must be a positive integer", Some("max_tokens")))?
            as usize,
    };
    let stop = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.iter().all(|s| s.is_string()) => a.iter().map(|s| s.as_str().unwrap().to_string()).collect(),
        _ => return Err(ApiError::bad("`stop` must be a string or a list of strings", Some("stop"))),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    let seed = match body.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_i64().ok_or_else(|| ApiError::bad("`seed` must be an integer", Some("seed")))? as u64),
    };
    if !matches!(body.get("n"), None | Some(Value::Null)) && body.get("n").and_then(|n| n.as_u64()) != Some(1) {
        return Err(ApiError::bad("only n=1 is supported", Some("n")));
    }
    // the tool-call grammar's per-request switch; the grammar itself is off in the serving
    // profile (DSV41_TOOL_GRAMMAR=0) and not ported, but the field is checked the same way
    if body.get("tool_grammar").is_some_and(|t| !t.is_boolean()) {
        return Err(ApiError::bad("`tool_grammar` must be a boolean", Some("tool_grammar")));
    }
    let flag = |k: &'static str| -> Result<bool, ApiError> {
        match body.get(k) {
            None | Some(Value::Null) => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            _ => Err(ApiError::bad(format!("`{k}` must be a boolean"), Some(k))),
        }
    };
    Ok(ChatReq {
        messages,
        thinking,
        effort,
        max_tokens,
        temperature: num(&body, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0)? as f32,
        top_p: num(&body, "top_p", DEFAULT_TOP_P, 0.0, 1.0)? as f32,
        stop,
        seed,
        ignore_eos: flag("ignore_eos")?,
        stream: flag("stream")?,
        include_usage: body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|b| b.as_bool()).unwrap_or(false),
    })
}

/// The prompt, its ids with every image placeholder expanded into its span, and the images.
fn render(req: &ChatReq, ctx: &Ctx) -> Result<(String, crate::vision::Expanded), ApiError> {
    let enc = encode_messages(&req.messages, req.thinking, req.effort)
        .map_err(|e| ApiError::bad(format!("cannot encode messages: {e}"), Some("messages")))?;
    let ids = ctx.text.encode(&enc.prompt).map_err(|e| ApiError { status: 500, message: format!("tokenizer: {e}"), param: None })?;
    let ex = match &ctx.vision {
        Some(v) => crate::vision::expand(&ids, &enc.images, v)
            .map_err(|e| ApiError::bad(format!("cannot process image inputs: {e}"), Some("messages")))?,
        None if !enc.images.is_empty() => return Err(ApiError::bad(NO_VISION, Some("messages"))),
        None => crate::vision::Expanded { ids, images: Vec::new(), key: None },
    };
    Ok((enc.prompt, ex))
}

pub fn debug_prompt(stream: &mut TcpStream, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    let req = parse(body, ctx)?;
    let (prompt, ex) = render(&req, ctx)?;
    send_json(stream, 200, &json!({"prompt": prompt, "prompt_tokens": ex.ids.len(), "ids": ex.ids, "thinking": req.thinking,
                                  "effort": req.effort, "images": ex.images.len()}))?;
    Ok(())
}

/// Incremental detokenization (the usual prefix/read offsets): text is released once it no
/// longer ends in a partial UTF-8 character.
#[derive(Default)]
struct Detok {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    fn push(&mut self, text: &Text, t: u32) -> String {
        self.ids.push(t);
        let prev = text.decode(&self.ids[self.prefix..self.read]).unwrap_or_default();
        let full = text.decode(&self.ids[self.prefix..]).unwrap_or_default();
        if full.len() > prev.len() && !full.ends_with('\u{FFFD}') && full.starts_with(&prev) {
            let delta = full[prev.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            delta
        } else {
            String::new()
        }
    }

    /// The rest, even a partial character (the answer is over).
    fn flush(&mut self, text: &Text) -> String {
        let full = text.decode(&self.ids[self.prefix..]).unwrap_or_default();
        let prev = text.decode(&self.ids[self.prefix..self.read]).unwrap_or_default();
        (self.prefix, self.read) = (self.ids.len(), self.ids.len());
        if full.len() > prev.len() { full.get(prev.len()..).unwrap_or("").to_string() } else { String::new() }
    }
}

/// One chunk's delta per run of a phase: {"reasoning_content": ...} or {"content": ...}.
fn deltas(pieces: Vec<(Phase, String)>) -> Vec<Value> {
    let mut runs: Vec<(Phase, String)> = Vec::new();
    for (ph, t) in pieces {
        match runs.last_mut() {
            Some((p, s)) if *p == ph => s.push_str(&t),
            _ => runs.push((ph, t)),
        }
    }
    runs.into_iter()
        .map(|(ph, t)| if ph == Phase::Reasoning { json!({"reasoning_content": t}) } else { json!({"content": t}) })
        .collect()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn chat(stream: &mut TcpStream, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    let t0 = Instant::now();
    let req = parse(body, ctx)?;
    let (_, ex) = render(&req, ctx)?;
    let (ids, images, key) = (ex.ids, ex.images, ex.key);
    // the drafter writes a block past the last position: keep its width of headroom
    let room = ctx.max_seq.saturating_sub(ids.len() + 8);
    if room == 0 {
        return Err(ApiError::bad(format!("the prompt is {} tokens; the context holds {}", ids.len(), ctx.max_seq), Some("messages")).into());
    }
    let max_tokens = req.max_tokens.min(room);
    let id = ctx.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = mpsc::channel::<Event>();
    let seed = req.seed.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(id));
    let job = Job {
        id,
        prompt: ids.clone(),
        max_tokens,
        sampling: Sampling { temperature: req.temperature, top_p: req.top_p },
        seed,
        stop: if req.ignore_eos { Vec::new() } else { vec![ctx.eos] },
        sink: Box::new(move |ev| {
            let _ = tx.send(ev);
        }),
        images,
        key,
    };
    ctx.engine.send(Cmd::Submit(job)).map_err(|_| ApiError { status: 503, message: "the engine is not running".into(), param: None })?;
    let cid = format!("chatcmpl-{id:x}{:x}", seed & 0xffff_ffff);
    let created = now();
    let chunk = |delta: Value, finish: Value| {
        json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": ctx.model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]})
    };
    if req.stream {
        start_events(stream)?;
        send_event(stream, &chunk(json!({"role": "assistant", "content": ""}), Value::Null).to_string())?;
    }
    let mut router = OutputRouter::new(req.thinking, req.stop.clone());
    let mut detok = Detok::default();
    let (mut first_token_s, mut hit_eos) = (None, false);
    let (finish, gen) = loop {
        let ev = match rx.recv() {
            Ok(ev) => ev,
            Err(_) => return Err(ApiError { status: 500, message: "the engine dropped the request".into(), param: None }.into()),
        };
        match ev {
            Event::Tokens(toks) => {
                first_token_s.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                let mut pieces = Vec::new();
                for t in toks {
                    if hit_eos || router.stopped {
                        break;
                    }
                    if t == ctx.eos {
                        hit_eos = !req.ignore_eos;
                        continue;
                    }
                    pieces.extend(router.feed(&detok.push(&ctx.text, t)));
                    if router.stopped {
                        // a stop string: the answer ends before it
                        let _ = ctx.engine.send(Cmd::Cancel(id));
                    }
                }
                if req.stream {
                    for d in deltas(pieces) {
                        if send_event(stream, &chunk(d, Value::Null).to_string()).is_err() {
                            // the client left: free the lane at the next step
                            let _ = ctx.engine.send(Cmd::Cancel(id));
                            return Ok(());
                        }
                    }
                }
            }
            Event::Done(Finish::Error(e), _) => {
                // the request never started (its images could not be encoded)
                let err = json!({"error": {"message": format!("cannot process image inputs: {e}"), "type": "server_error",
                                           "param": null, "code": null}});
                if req.stream {
                    send_event(stream, &err.to_string())?;
                    send_event(stream, "[DONE]")?;
                    return Ok(());
                }
                return Err(ApiError { status: 500, message: format!("cannot process image inputs: {e}"), param: None }.into());
            }
            Event::Done(f, g) => break (f, g),
        }
    };
    let mut tail = router.feed(&detok.flush(&ctx.text));
    tail.extend(router.finish());
    let (calls, unparsed) = router.tool_calls(req.thinking);
    let finish_reason = if !calls.is_empty() {
        "tool_calls"
    } else if hit_eos || router.stopped {
        "stop"
    } else {
        match finish {
            Finish::Length => "length",
            _ => "stop",
        }
    };
    let n = gen.tokens.len();
    let reasoning_tokens = if req.thinking {
        gen.tokens.iter().position(|&t| Some(t) == ctx.think_end).unwrap_or(n)
    } else {
        0
    };
    let decode_s = (t0.elapsed().as_secs_f64() - first_token_s.unwrap_or(0.0)).max(1e-9);
    let mut stats = json!({
        "steps": gen.steps,
        "accept_len_mean": if gen.steps > 0 { (n.saturating_sub(1)) as f64 / gen.steps as f64 } else { 0.0 },
        "decode_tok_s": (n.saturating_sub(1)) as f64 / decode_s,
        "ttft_s": first_token_s.unwrap_or(0.0),
        "engine": "tidelm-rust",
    });
    // adaptive residency: the Python engine's prune_miss_request, plus each phase's rate
    let m = &gen.miss;
    let (missed, picks) = (m[0] + m[4], m[1] + m[5]);
    if picks > 0.0 {
        let rate = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
        stats["prune_miss_request"] = json!({
            "miss_rate": rate(missed, picks), "missed_slots": missed as u64, "slots": picks as u64,
            "score_miss_rate": rate(m[2] + m[6], m[3] + m[7]),
            "prompt_miss_rate": rate(m[0], m[1]), "decode_miss_rate": rate(m[4], m[5]),
        });
    }
    let usage = json!({
        "prompt_tokens": ids.len(),
        "completion_tokens": n,
        "total_tokens": ids.len() + n,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "prompt_tokens_details": {"cached_tokens": gen.cached},
    });
    if req.stream {
        // a block that did not parse goes out as content, as the non-streamed answer has it
        // (the Python server's stream drops it)
        let mut last = deltas(tail);
        if !unparsed.is_empty() {
            last.push(json!({"content": unparsed}));
        }
        if !calls.is_empty() {
            let indexed: Vec<Value> = calls.iter().enumerate().map(|(i, c)| {
                let mut c = c.clone();
                c["index"] = json!(i);
                c
            }).collect();
            last.push(json!({"tool_calls": indexed}));
        }
        for d in last {
            send_event(stream, &chunk(d, Value::Null).to_string())?;
        }
        let mut end = chunk(json!({}), Value::String(finish_reason.into()));
        end["usage"] = usage.clone();
        end["x_engine_stats"] = stats;
        send_event(stream, &end.to_string())?;
        if req.include_usage {
            send_event(stream, &json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": ctx.model,
                                       "choices": [], "usage": usage}).to_string())?;
        }
        send_event(stream, "[DONE]")?;
        return Ok(());
    }
    let mut message = json!({"role": "assistant", "content": router.content});
    if req.thinking {
        message["reasoning_content"] = Value::String(router.reasoning.clone());
    }
    if !calls.is_empty() {
        if router.content.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = Value::Array(calls);
    }
    send_json(stream, 200, &json!({
        "id": cid, "object": "chat.completion", "created": created, "model": ctx.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": null}],
        "usage": usage, "x_engine_stats": stats,
    }))?;
    Ok(())
}
