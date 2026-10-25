//! POST /v1/chat/completions and /v1/debug/prompt: server/app.py's request rules
//! (parse_sampling, resolve_thinking, build_chat_prompt) over the Rust chat encoder, streamed
//! as OpenAI chunks with usage.

use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tidelm_engine::decode::{Event, Finish, Job};
use tidelm_engine::forward::Sampling;
use tidelm_text::chat::{encode, Message, Role};
use tidelm_text::Text;

use crate::http::{send_event, send_json, start_events, ApiError};
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
    messages: Vec<Message>,
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

/// A message's text: a string, or the text parts of a content list.
fn content_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    Some(t) => return Err(ApiError::bad(format!("content parts of type {t:?} are not supported"), Some("messages"))),
                    None => return Err(ApiError::bad("a content part needs a `type`", Some("messages"))),
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad("`content` must be a string or a list of parts", Some("messages"))),
    }
}

fn parse_messages(body: &Value) -> Result<Vec<Message>, ApiError> {
    let msgs = body.get("messages").and_then(|m| m.as_array()).filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("`messages` must be a non-empty list", Some("messages")))?;
    if body.get("tools").is_some_and(|t| !t.is_null() && t.as_array().is_none_or(|a| !a.is_empty())) {
        return Err(ApiError::bad("tools are not supported by the Rust server yet", Some("tools")));
    }
    msgs.iter()
        .enumerate()
        .map(|(i, m)| {
            let role = match m.get("role").and_then(|r| r.as_str()) {
                // V4.1 has no developer role: fold it into system, as server/app.py does
                Some("system") | Some("developer") => Role::System,
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                Some(r) => return Err(ApiError::bad(format!("messages[{i}]: role {r:?} is not supported"), Some("messages"))),
                None => return Err(ApiError::bad(format!("messages[{i}] must be an object with a `role`"), Some("messages"))),
            };
            if m.get("tool_calls").is_some_and(|t| !t.is_null()) {
                return Err(ApiError::bad("tool calls are not supported by the Rust server yet", Some("messages")));
            }
            Ok(Message {
                role,
                content: content_text(m.get("content").unwrap_or(&Value::Null))?,
                reasoning: m.get("reasoning_content").and_then(|r| r.as_str()).map(String::from),
            })
        })
        .collect()
}

fn parse(body: &[u8], ctx: &Ctx) -> Result<ChatReq, ApiError> {
    let body: Value = serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid JSON: {e}"), None))?;
    let messages = parse_messages(&body)?;
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

fn render(req: &ChatReq, text: &Text) -> Result<(String, Vec<u32>), ApiError> {
    let prompt = encode(&req.messages, req.thinking, Some(req.effort)).map_err(|e| ApiError::bad(e, Some("messages")))?;
    let ids = text.encode(&prompt).map_err(|e| ApiError { status: 500, message: format!("tokenizer: {e}"), param: None })?;
    Ok((prompt, ids))
}

pub fn debug_prompt(stream: &mut TcpStream, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    let req = parse(body, ctx)?;
    let (prompt, ids) = render(&req, &ctx.text)?;
    send_json(stream, 200, &json!({"prompt": prompt, "prompt_tokens": ids.len(), "ids": ids, "thinking": req.thinking,
                                  "effort": req.effort}))?;
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
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn chat(stream: &mut TcpStream, body: &[u8], ctx: &Ctx) -> Result<(), Failure> {
    let t0 = Instant::now();
    let req = parse(body, ctx)?;
    let (_, ids) = render(&req, &ctx.text)?;
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
    };
    ctx.engine.send(Cmd::Submit(job)).map_err(|_| ApiError { status: 503, message: "the engine is not running".into(), param: None })?;
    let cid = format!("chatcmpl-{id:x}{:x}", seed & 0xffff_ffff);
    let created = now();
    let chunk = |delta: Value, finish: Value| {
        json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": ctx.model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        .to_string()
    };
    if req.stream {
        start_events(stream)?;
        send_event(stream, &chunk(json!({"role": "assistant", "content": ""}), Value::Null))?;
    }
    // the answer so far, split at the end of thinking
    let (mut reasoning, mut content) = (String::new(), String::new());
    let (mut dr, mut dc) = (Detok::default(), Detok::default());
    let mut in_thinking = req.thinking;
    let (mut reasoning_tokens, mut first_token_s) = (0usize, None);
    let mut stopped_by_text = false;
    let (finish, gen) = loop {
        let ev = match rx.recv() {
            Ok(ev) => ev,
            Err(_) => return Err(ApiError { status: 500, message: "the engine dropped the request".into(), param: None }.into()),
        };
        match ev {
            Event::Tokens(toks) => {
                first_token_s.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                let (mut dreason, mut dcontent) = (String::new(), String::new());
                for t in toks {
                    if stopped_by_text || t == ctx.eos {
                        continue;
                    }
                    if in_thinking {
                        reasoning_tokens += 1;
                        if Some(t) == ctx.think_end {
                            in_thinking = false;
                        } else {
                            dreason.push_str(&dr.push(&ctx.text, t));
                        }
                        continue;
                    }
                    dcontent.push_str(&dc.push(&ctx.text, t));
                    // text stop sequences end the answer just before themselves; a match can
                    // only start in the last (longest stop - 1) bytes of what came before
                    if !req.stop.is_empty() {
                        let longest = req.stop.iter().map(|s| s.len()).max().unwrap_or(0);
                        let mut from = content.len().saturating_sub(longest);
                        while !content.is_char_boundary(from) {
                            from -= 1;
                        }
                        let tail = format!("{}{dcontent}", &content[from..]);
                        if let Some(at) = req.stop.iter().filter_map(|s| tail.find(s.as_str())).min() {
                            // the stop may begin in text already sent; only what follows is cut
                            let keep = (from + at).saturating_sub(content.len());
                            dcontent.truncate(keep.min(dcontent.len()));
                            stopped_by_text = true;
                            let _ = ctx.engine.send(Cmd::Cancel(id));
                        }
                    }
                }
                reasoning.push_str(&dreason);
                content.push_str(&dcontent);
                if req.stream && (!dreason.is_empty() || !dcontent.is_empty()) {
                    let mut delta = serde_json::Map::new();
                    if !dreason.is_empty() {
                        delta.insert("reasoning_content".into(), Value::String(dreason));
                    }
                    if !dcontent.is_empty() {
                        delta.insert("content".into(), Value::String(dcontent));
                    }
                    if send_event(stream, &chunk(Value::Object(delta), Value::Null)).is_err() {
                        // the client left: free the lane at the next step
                        let _ = ctx.engine.send(Cmd::Cancel(id));
                        return Ok(());
                    }
                }
            }
            Event::Done(f, g) => break (f, g),
        }
    };
    let finish_reason = match finish {
        Finish::Length if !stopped_by_text => "length",
        _ => "stop",
    };
    let n = gen.tokens.len();
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
    });
    if req.stream {
        let mut last: Value = serde_json::from_str(&chunk(json!({}), Value::String(finish_reason.into()))).unwrap();
        last["x_engine_stats"] = stats;
        send_event(stream, &last.to_string())?;
        if req.include_usage {
            send_event(stream, &json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": ctx.model,
                                       "choices": [], "usage": usage}).to_string())?;
        }
        send_event(stream, "[DONE]")?;
        return Ok(());
    }
    let mut message = json!({"role": "assistant", "content": content});
    if req.thinking {
        message["reasoning_content"] = Value::String(reasoning);
    }
    send_json(stream, 200, &json!({
        "id": cid, "object": "chat.completion", "created": created, "model": ctx.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage, "x_engine_stats": stats,
    }))?;
    Ok(())
}
