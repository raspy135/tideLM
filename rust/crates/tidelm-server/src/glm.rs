//! GLM-5.3's routes: server/muse_app.py's request rules and response shapes over
//! tidelm_text::glm (server/glm_app.py's template and parser).
//!
//! - prompt: the request's messages and tools through the checkpoint's template, at the
//!   request's effort (reasoning.effort, reasoning_effort, chat_template_kwargs.reasoning_effort,
//!   else GLM_REASONING_EFFORT); the template honors low and high, anything else is Max, and
//!   thinking cannot be turned off.
//! - params: max_tokens || max_completion_tokens || GLM_DEFAULT_MAX_TOKENS, clamped to the
//!   context (max_seq - prompt - 1; a prompt of max_seq is refused); temperature and top_p from
//!   the request else generation_config.json; seed from the request else the clock. The stop
//!   ids are generation_config's eos ids. stop, ignore_eos, n, tool_choice, response_format,
//!   logprobs, stream_options and top_k are accepted and not used (the engine's sampler has no
//!   top_k; generation_config has none either).
//! - answers: reasoning_content, content and tool_calls (parsed once the answer is done), usage
//!   with the engine's stats under `muse`. Streamed: one chunk per routed delta, a `: keepalive`
//!   comment after a second without one while tokens flow, all tool calls in one chunk at the
//!   end, usage and the stats on the final chunk.

use std::io::Write;
use std::net::TcpStream;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tidelm_engine::decode::{Event, Finish, Generation};
use tidelm_engine::forward::Sampling;
use tidelm_text::chat::py_json;
use tidelm_text::glm::{render, truthy, GlmIds};
use tidelm_text::Text;

use crate::anthropic::{Prep, Prepared, Room};
use crate::api::{body_json, submit, Failure, Request};
use crate::http::{send_json, start_events, ApiError};
use crate::output::{hex24, Answer, GlmAnswer, Phase};
use crate::{Cmd, Ctx};

/// What a GLM server needs beyond the tokenizer: the parser's ids and muse_app's defaults.
pub struct GlmSettings {
    /// `<think>`, `</think>`, `<tool_call>`, `</tool_call>` and the stop ids
    pub ids: GlmIds,
    /// GLM_REASONING_EFFORT (muse_app's default "medium", which the template renders as Max)
    pub default_effort: String,
    /// GLM_DEFAULT_MAX_TOKENS (4096)
    pub default_max_tokens: i64,
    /// generation_config.json's temperature and top_p (1.0 when absent)
    pub temperature: f64,
    pub top_p: f64,
}

impl GlmSettings {
    pub fn load(model_dir: &Path, text: &Text) -> Result<Self, tidelm_text::Error> {
        let ids = GlmIds::load(model_dir, text)?;
        let path = model_dir.join("generation_config.json");
        let gen: Value = if path.exists() { serde_json::from_slice(&std::fs::read(path)?)? } else { Value::Null };
        let f = |k: &str| gen.get(k).and_then(|v| v.as_f64()).unwrap_or(1.0);
        Ok(GlmSettings {
            ids,
            default_effort: std::env::var("GLM_REASONING_EFFORT").unwrap_or_else(|_| "medium".into()),
            default_max_tokens: match std::env::var("GLM_DEFAULT_MAX_TOKENS") {
                Ok(v) => v.parse().map_err(|_| format!("GLM_DEFAULT_MAX_TOKENS={v}: not an integer"))?,
                Err(_) => 4096,
            },
            temperature: f("temperature"),
            top_p: f("top_p"),
        })
    }
}

/// request_effort: the first of reasoning.effort, reasoning_effort and
/// chat_template_kwargs.reasoning_effort that is neither null nor ""; None: the server's.
pub fn request_effort(body: &Value) -> Option<&Value> {
    let nested = |k: &str| body.get(k).filter(|v| v.is_object());
    [nested("reasoning").and_then(|r| r.get("effort")), body.get("reasoning_effort"),
     nested("chat_template_kwargs").and_then(|c| c.get("reasoning_effort"))]
        .into_iter()
        .flatten()
        .find(|v| !v.is_null() && v.as_str() != Some(""))
}

/// The effort the template sees: a request's non-string effort is none (Max)
fn template_effort<'a>(body: &'a Value, g: &'a GlmSettings) -> Option<&'a str> {
    match request_effort(body) {
        Some(v) => v.as_str(),
        None => Some(&g.default_effort),
    }
}

fn type_error(key: &str, v: &Value) -> ApiError {
    ApiError::bad(format!("`{key}`: {v} is not a number"), None)
}

/// `int(v)`
fn py_int(key: &str, v: &Value) -> Result<i64, ApiError> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).ok_or_else(|| type_error(key, v)),
        Value::Bool(b) => Ok(*b as i64),
        Value::String(s) => s.trim().replace('_', "").parse().map_err(|_| type_error(key, v)),
        _ => Err(type_error(key, v)),
    }
}

/// `float(v)`
fn py_float(key: &str, v: &Value) -> Result<f64, ApiError> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| type_error(key, v)),
        Value::Bool(b) => Ok(*b as i64 as f64),
        Value::String(s) => s.trim().parse().map_err(|_| type_error(key, v)),
        _ => Err(type_error(key, v)),
    }
}

pub struct Params {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub copy_draft: bool,
}

/// muse_app's params. Beyond Python's: max_tokens below 1, a negative or non-finite
/// temperature and a top_p outside [0, 1] are refused rather than handed to the sampler.
pub fn params(body: &Value, g: &GlmSettings) -> Result<Params, ApiError> {
    let mt = ["max_tokens", "max_completion_tokens"].into_iter().find_map(|k| body.get(k).filter(|v| truthy(v)).map(|v| (k, v)));
    let max_tokens = match mt {
        Some((k, v)) => py_int(k, v)?,
        None => g.default_max_tokens,
    };
    let temperature = body.get("temperature").map(|v| py_float("temperature", v)).transpose()?.unwrap_or(g.temperature);
    let top_p = body.get("top_p").map(|v| py_float("top_p", v)).transpose()?.unwrap_or(g.top_p);
    // int(top_k) refuses what Python's does; the value itself has no use here
    body.get("top_k").map(|v| py_int("top_k", v)).transpose()?;
    let seed = match body.get("seed") {
        Some(v) => py_int("seed", v)?,
        None => (SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0) % (1 << 31)) as i64,
    };
    if max_tokens < 1 {
        return Err(ApiError::bad(format!("`max_tokens` must be at least 1, not {max_tokens}"), Some("max_tokens")));
    }
    if !(temperature.is_finite() && temperature >= 0.0) {
        return Err(ApiError::bad(format!("`temperature` must be a finite number >= 0, not {temperature}"), Some("temperature")));
    }
    if !(0.0..=1.0).contains(&top_p) {
        return Err(ApiError::bad(format!("`top_p` must be in [0, 1], not {top_p}"), Some("top_p")));
    }
    Ok(Params {
        max_tokens: max_tokens as usize,
        temperature: temperature as f32,
        top_p: top_p as f32,
        seed: seed as u64,
        copy_draft: !matches!(body.get("x_copy_draft"), Some(Value::Bool(false))),
    })
}

fn encode(ctx: &Ctx, text: &str) -> Result<Vec<u32>, ApiError> {
    ctx.text.encode(text).map_err(|e| ApiError { status: 500, message: format!("tokenizer: {e}"), param: None })
}

fn object(body: Value) -> Result<Value, ApiError> {
    if body.is_object() {
        Ok(body)
    } else {
        Err(ApiError::bad("the request must be a JSON object", None))
    }
}

/// The prompt of a chat body: its messages (none: []) and tools at its effort
fn prompt(body: &Value, ctx: &Ctx, effort: Option<&str>) -> Result<(String, Vec<u32>), ApiError> {
    let msgs: &[Value] = match body.get("messages") {
        None => &[],
        Some(Value::Array(a)) => a,
        Some(_) => return Err(ApiError::bad("`messages` must be a list", Some("messages"))),
    };
    let text = render(msgs, body.get("tools"), effort).map_err(|e| ApiError::bad(format!("cannot render messages: {e}"), Some("messages")))?;
    let ids = encode(ctx, &text)?;
    Ok((text, ids))
}

fn request(prompt: Vec<u32>, p: &Params, g: &GlmSettings) -> Request {
    Request {
        prompt,
        images: Vec::new(),
        key: None,
        max_tokens: p.max_tokens,
        sampling: Sampling { temperature: p.temperature, top_p: p.top_p },
        seed: p.seed,
        stop: g.ids.stop.clone(),
        copy_draft: p.copy_draft,
    }
}

fn exceeds(n: usize) -> ApiError {
    ApiError::bad(format!("prompt of {n} tokens exceeds the context"), Some("messages"))
}

/// /v1/debug/prompt: the rendered prompt and its ids, no engine work (bench/bench.py sizes its
/// prompts with it)
pub fn debug_prompt(stream: &mut TcpStream, body: &[u8], ctx: &Ctx, g: &GlmSettings) -> Result<(), Failure> {
    let body = object(body_json(body)?)?;
    let effort = template_effort(&body, g);
    let (text, ids) = prompt(&body, ctx, effort)?;
    send_json(stream, 200, &json!({"prompt_tokens": ids.len(), "prompt": text, "ids": ids, "effort": effort}))?;
    Ok(())
}

pub fn chat(stream: &mut TcpStream, body: &[u8], ctx: &Ctx, g: &GlmSettings) -> Result<(), Failure> {
    let t0 = Instant::now();
    let body = object(body_json(body)?)?;
    let (_, ids) = prompt(&body, ctx, template_effort(&body, g))?;
    let mut p = params(&body, g)?;
    p.max_tokens = Room::Glm.fit(ids.len(), p.max_tokens, ctx.max_seq).ok_or_else(|| exceeds(ids.len()))?;
    let tools = body.get("tools").filter(|t| !t.is_null());
    run(stream, ctx, g, request(ids, &p, g), tools, body.get("stream").is_some_and(truthy), true, t0)
}

/// /v1/completions: a raw prompt; `text` is the decode of every answer id, the stop id too
pub fn completions(stream: &mut TcpStream, body: &[u8], ctx: &Ctx, g: &GlmSettings) -> Result<(), Failure> {
    let t0 = Instant::now();
    let body = object(body_json(body)?)?;
    let text = match body.get("prompt") {
        None => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return Err(ApiError::bad("`prompt` must be a string", Some("prompt")).into()),
    };
    let ids = encode(ctx, text)?;
    if ids.is_empty() {
        return Err(ApiError::bad("the prompt is empty", Some("prompt")).into());
    }
    let mut p = params(&body, g)?;
    p.max_tokens = Room::Glm.fit(ids.len(), p.max_tokens, ctx.max_seq).ok_or_else(|| exceeds(ids.len()))?;
    run(stream, ctx, g, request(ids, &p, g), None, body.get("stream").is_some_and(truthy), false, t0)
}

/// /v1/messages on GLM: the translated request rendered at the server's effort, muse_app's
/// params from the Anthropic body
pub fn anthropic_prepare(req: &Value, msgs: &[Value], tools: Vec<Value>, ctx: &Ctx, g: &GlmSettings, count: bool) -> Result<Prep, ApiError> {
    let tools = Value::Array(tools);
    let text = render(msgs, Some(&tools), Some(&g.default_effort)).map_err(|e| ApiError::bad(format!("cannot render messages: {e}"), Some("messages")))?;
    let ids = encode(ctx, &text)?;
    if count {
        return Ok(Prep::Count(ids.len()));
    }
    let p = params(req, g)?;
    Ok(Prep::Run(Prepared { request: request(ids, &p, g), answer: Box::new(GlmAnswer::new(&g.ids)), room: Room::Glm, tools }))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// muse_app's job.stats, from the Rust engine's Generation
fn stats(prompt: usize, n: usize, gen: &Generation, first: Option<f64>, total: f64) -> Value {
    let round = |x: f64, d: i32| (x * 10f64.powi(d)).round() / 10f64.powi(d);
    let m = &gen.miss;
    let rate = |a: f64, b: f64| if b > 0.0 { round(a / b, 4) } else { 0.0 };
    json!({
        "prompt_tokens": prompt,
        "reused_prefix": gen.cached,
        "completion_tokens": n,
        "ttft_s": round(first.unwrap_or(total), 3),
        "decode_tok_s": match first {
            Some(f) if n > 1 && total > f => json!(round((n - 1) as f64 / (total - f), 2)),
            _ => Value::Null,
        },
        "engine": "tidelm-rust",
        "steps": gen.steps,
        "accept_len_mean": if gen.steps > 0 { n.saturating_sub(1) as f64 / gen.steps as f64 } else { 0.0 },
        "copy_steps": gen.copy_steps,
        "copy_accepted": gen.copy_accepted,
        // adaptive residency: DS's prune_miss_request, the same fields (api.rs)
        "prune_miss_request": {
            "miss_rate": rate(m[0] + m[4], m[1] + m[5]), "missed_slots": (m[0] + m[4]) as u64, "slots": (m[1] + m[5]) as u64,
            "score_miss_rate": rate(m[2] + m[6], m[3] + m[7]),
            "prompt_miss_rate": rate(m[0], m[1]), "decode_miss_rate": rate(m[4], m[5]),
        },
    })
}

/// tool_calls when there are calls; length when the answer ran out of tokens before a stop
/// id; else stop
fn finish_reason(calls: &[Value], answer: &dyn Answer, finish: &Finish) -> &'static str {
    if !calls.is_empty() {
        "tool_calls"
    } else if !answer.ended() && matches!(finish, Finish::Length) {
        "length"
    } else {
        "stop"
    }
}

fn delta(ph: Phase, s: &str) -> Value {
    if ph == Phase::Reasoning { json!({"reasoning_content": s}) } else { json!({"content": s}) }
}

/// An SSE stream of `data:` lines in json.dumps(ensure_ascii=False)'s bytes
struct Sse<'a> {
    w: &'a mut TcpStream,
    last: Instant,
}

impl Sse<'_> {
    fn send(&mut self, v: &Value) -> std::io::Result<()> {
        write!(self.w, "data: {}\n\n", py_json(v))?;
        self.w.flush()?;
        self.last = Instant::now();
        Ok(())
    }

    /// Reasoning and tool-call bodies stream nothing visible; a comment keeps proxies awake and
    /// is how a closed client gets noticed.
    fn keepalive(&mut self) -> std::io::Result<()> {
        if self.last.elapsed() > Duration::from_secs(1) {
            self.w.write_all(b": keepalive\n\n")?;
            self.w.flush()?;
            self.last = Instant::now();
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run(stream: &mut TcpStream, ctx: &Ctx, g: &GlmSettings, req: Request, tools: Option<&Value>, streaming: bool, chat: bool,
       t0: Instant) -> Result<(), Failure> {
    let n_prompt = req.prompt.len();
    let mut answer = GlmAnswer::new(&g.ids);
    // the answer's ids, through the stop id (/v1/completions' text)
    let mut ids = Vec::new();
    let mut first = None;
    let (id, rx) = submit(ctx, req)?;
    let rid = format!("chatcmpl-{}", hex24());
    let created = now();
    let chunk = |delta: Value, finish: Value| {
        json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": ctx.model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let gone = |_: std::io::Error| {
        let _ = ctx.engine.send(Cmd::Cancel(id));
        Failure::Io
    };
    let mut sse = None;
    if streaming {
        start_events(stream)?;
        let mut s = Sse { w: &mut *stream, last: Instant::now() };
        s.send(&chunk(json!({"role": "assistant"}), Value::Null)).map_err(gone)?;
        sse = Some(s);
    }
    let (finish, gen) = loop {
        match rx.recv() {
            Ok(Event::Tokens(t)) => {
                first.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                let before = answer.tokens();
                let events = answer.feed(&ctx.text, &t);
                ids.extend_from_slice(&t[..answer.tokens() - before]);
                if let Some(s) = sse.as_mut() {
                    for (ph, d) in &events {
                        s.send(&chunk(delta(*ph, d), Value::Null)).map_err(gone)?;
                    }
                    if events.is_empty() {
                        s.keepalive().map_err(gone)?;
                    }
                }
            }
            Ok(Event::Done(Finish::Error(e), _)) => {
                if let Some(s) = sse.as_mut() {
                    let _ = s.send(&json!({"error": {"message": e, "type": "server_error", "param": null, "code": null}}));
                    return Ok(());
                }
                return Err(ApiError { status: 500, message: e, param: None }.into());
            }
            Ok(Event::Done(f, gen)) => break (f, gen),
            Err(_) => return Err(ApiError { status: 500, message: "the engine dropped the request".into(), param: None }.into()),
        }
    };
    let tail = answer.finish(&ctx.text);
    let total = t0.elapsed().as_secs_f64();
    let n = answer.tokens();
    let st = stats(n_prompt, n, &gen, first, total);
    let usage = json!({"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n, "muse": st});
    let (calls, _) = answer.tool_calls(&ctx.text, tools);
    let reason = finish_reason(&calls, &answer, &finish);
    if let Some(mut s) = sse {
        for (ph, d) in &tail {
            s.send(&chunk(delta(*ph, d), Value::Null)).map_err(gone)?;
        }
        if !calls.is_empty() {
            let indexed: Vec<Value> = calls.into_iter().enumerate().map(|(i, mut c)| {
                c["index"] = json!(i);
                c
            }).collect();
            s.send(&chunk(json!({"tool_calls": indexed}), Value::Null)).map_err(gone)?;
        }
        // usage and the engine's stats ride the final chunk: a streamed client (bench/bench.py)
        // otherwise has no token count to divide by
        let mut end = chunk(json!({}), json!(reason));
        end["usage"] = usage;
        end["x_engine_stats"] = st;
        s.send(&end).map_err(gone)?;
        s.w.write_all(b"data: [DONE]\n\n").map_err(gone)?;
        s.w.flush().map_err(gone)?;
        return Ok(());
    }
    if !chat {
        send_json(stream, 200, &json!({"id": rid, "object": "text_completion", "model": ctx.model,
                                       "choices": [{"index": 0, "text": ctx.text.decode(&ids).unwrap_or_default(), "finish_reason": "stop"}],
                                       "usage": usage}))?;
        return Ok(());
    }
    let content = answer.content();
    let mut msg = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }});
    if !answer.reasoning().is_empty() {
        msg["reasoning_content"] = json!(answer.reasoning());
    }
    if !calls.is_empty() {
        msg["tool_calls"] = Value::Array(calls);
    }
    send_json(stream, 200, &json!({"id": rid, "object": "chat.completion", "created": created, "model": ctx.model,
                                   "choices": [{"index": 0, "message": msg, "finish_reason": reason}], "usage": usage}))?;
    Ok(())
}
