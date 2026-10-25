//! The routes over HTTP against a fake engine thread that replays a canned answer the way the
//! scheduler emits one (steps of a few tokens, accepted drafts past a stop id included): GLM's
//! OpenAI routes, and /v1/messages on both families - server/test_anthropic_api.py's HttpTests
//! and server/test_glm_app.py's round trip. No GPU. Each needs its checkpoint's tokenizer
//! (GLM_TOKENIZER_DIR, default ~/models/GLM-5.3-EXL3-3.0bpw; TIDELM_MODEL_DIR, default
//! ~/models/DeepSeek-V4.1-Flash-1017) and skips without it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc, Mutex};

use serde_json::{json, Value};
use tidelm_engine::decode::{Event, Finish, Generation};
use tidelm_text::glm::{render, GlmIds};
use tidelm_text::Text;

use crate::anthropic::{to_chat, ReasoningMemo};
use crate::glm::{request_effort, GlmSettings};
use crate::{Cmd, Ctx, Family};

fn dir(var: &str, default: &str) -> Option<PathBuf> {
    let d = std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(default));
    d.join("tokenizer.json").exists().then_some(d)
}

/// What the fake engine saw and what it answers.
#[derive(Default)]
struct Engine {
    jobs: Mutex<Vec<(Vec<u32>, usize, Vec<u32>)>>,
    answer: Mutex<Vec<u32>>,
}

impl Engine {
    fn last(&self) -> (Vec<u32>, usize, Vec<u32>) {
        self.jobs.lock().unwrap().last().cloned().unwrap()
    }
}

/// A server on 127.0.0.1 with `family`; the engine thread answers every job with
/// `engine.answer`, three tokens a step, reusing 7 prompt tokens.
fn serve(text: Text, family: Family, eos: u32, max_seq: usize) -> (String, Arc<Engine>, Arc<Ctx>) {
    let (tx, rx) = mpsc::channel::<Cmd>();
    let engine = Arc::new(Engine::default());
    let ctx = Arc::new(Ctx {
        text,
        model: "test-model".into(),
        max_seq,
        default_thinking: false,
        default_effort: 75,
        eos,
        think_end: None,
        vision: None,
        engine: tx,
        next_id: AtomicU64::new(1),
        family,
        memo: ReasoningMemo::default(),
    });
    let e = engine.clone();
    std::thread::spawn(move || {
        for cmd in rx {
            let Cmd::Submit(mut job) = cmd else { continue };
            e.jobs.lock().unwrap().push((job.prompt.clone(), job.max_tokens, job.stop.clone()));
            let answer = e.answer.lock().unwrap().clone();
            let mut gen = Generation { cached: 7, ..Default::default() };
            let mut finish = Finish::Length;
            for step in answer.chunks(3) {
                let step = &step[..step.len().min(job.max_tokens - gen.tokens.len())];
                gen.tokens.extend_from_slice(step);
                gen.steps += 1;
                (job.sink)(Event::Tokens(step.to_vec()));
                if step.iter().any(|t| job.stop.contains(t)) {
                    finish = Finish::Stop;
                    break;
                }
                if gen.tokens.len() >= job.max_tokens {
                    break;
                }
            }
            (job.sink)(Event::Done(finish, gen));
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let c = ctx.clone();
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let c = c.clone();
            std::thread::spawn(move || crate::http::serve(s, &c));
        }
    });
    (addr, engine, ctx)
}

/// (status, content type, body)
fn post(addr: &str, path: &str, body: &Value) -> (u16, String, String) {
    let body = body.to_string();
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let ctype = head.lines().find_map(|l| l.strip_prefix("Content-Type: ")).unwrap_or("").to_string();
    (status, ctype, body.to_string())
}

fn sse_data(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|c| c.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

fn glm() -> Option<(String, Arc<Engine>, Arc<Ctx>)> {
    let d = dir("GLM_TOKENIZER_DIR", "models/GLM-5.3-EXL3-3.0bpw")?;
    let text = Text::load(&d).unwrap();
    let ids = GlmIds::load(&d, &text).unwrap();
    let eos = ids.stop[0];
    let g = GlmSettings { ids, default_effort: "low".into(), default_max_tokens: 4096, temperature: 1.0, top_p: 1.0 };
    Some(serve(text, Family::Glm(g), eos, 65536))
}

/// server/test_anthropic_api.py's REQ
fn anthropic_req() -> Value {
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

fn kinds_and_events(body: &str) -> (Vec<String>, Vec<Value>) {
    let mut events = Vec::new();
    for chunk in body.trim().split("\n\n") {
        let (kind, data) = chunk.split_once('\n').unwrap();
        let v: Value = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(kind, format!("event: {}", v["type"].as_str().unwrap()));
        events.push(v);
    }
    let mut kinds: Vec<String> = events.iter().map(|e| e["type"].as_str().unwrap().to_string()).filter(|k| k != "ping").collect();
    kinds.dedup(); // text arrives in pieces
    (kinds, events)
}

#[test]
fn glm_messages_route() {
    let Some((addr, engine, ctx)) = glm() else {
        eprintln!("GLM tokenizer not present; skipping");
        return;
    };
    let answer = "I should list files.</think>\nLet me look.<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call><|observation|>";
    *engine.answer.lock().unwrap() = ctx.text.encode(answer).unwrap();
    let req = anthropic_req();

    // the stream: events in order, no reasoning, no leading newline, the call at index 1
    let mut streamed = req.clone();
    streamed["stream"] = json!(true);
    let (status, ctype, body) = post(&addr, "/v1/messages?beta=true", &streamed);
    assert_eq!((status, ctype.as_str()), (200, "text/event-stream"));
    let (kinds, events) = kinds_and_events(&body);
    assert_eq!(kinds, ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start",
                       "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
    let text: String = events.iter().filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
        .map(|e| e["delta"]["text"].as_str().unwrap()).collect();
    assert_eq!(text, "Let me look.");
    let uses: Vec<&Value> = events.iter().filter(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use").collect();
    assert_eq!(uses.len(), 1);
    assert_eq!((uses[0]["index"].as_u64(), uses[0]["content_block"]["name"].as_str()), (Some(1), Some("Bash")));
    let args: String = events.iter().filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "input_json_delta")
        .map(|e| e["delta"]["partial_json"].as_str().unwrap()).collect();
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"command": "ls -la"}));
    let delta = events.iter().find(|e| e["type"] == "message_delta").unwrap();
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(delta["usage"]["cache_read_input_tokens"], 7);
    let n_prompt = engine.last().0.len();
    assert_eq!(events[0]["message"]["usage"]["input_tokens"].as_u64(), Some(n_prompt as u64));
    assert_eq!(delta["usage"]["input_tokens"].as_u64(), Some(n_prompt as u64 - 7));
    // every answer token through <|observation|>
    assert_eq!(delta["usage"]["output_tokens"].as_u64(), Some(ctx.text.encode(answer).unwrap().len() as u64));

    // the JSON reply, and the reasoning comes back on the next turn
    let (status, _, body) = post(&addr, "/v1/messages", &req);
    assert_eq!(status, 200);
    let msg: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((msg["type"].as_str(), msg["stop_reason"].as_str()), (Some("message"), Some("tool_use")));
    assert_eq!(msg["content"][0], json!({"type": "text", "text": "Let me look."}));
    let usage = &msg["content"][1];
    assert_eq!((usage["type"].as_str(), usage["name"].as_str(), &usage["input"]), (Some("tool_use"), Some("Bash"), &json!({"command": "ls -la"})));
    assert!(usage["id"].as_str().unwrap().starts_with("toolu_"));
    let (first, max_tokens, stop) = engine.last();
    assert_eq!((max_tokens, stop), (512, vec![154820, 154827, 154829]));
    let first_text = ctx.text.decode(&first).unwrap();
    assert!(first_text.starts_with("[gMASK]<sop><|system|>Reasoning Effort: Low<|system|>\n# Tools"), "{first_text}");
    assert!(!first_text.contains("billing-header"));
    // The client sends the turn back without its reasoning; the prompt must still extend the
    // first one through the reasoning, which is what the engine's prefix cache matches.
    let mut follow = req.clone();
    follow["messages"].as_array_mut().unwrap().extend([
        json!({"role": "assistant", "content": msg["content"]}),
        json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": usage["id"], "content": "a.txt"}]})]);
    let (status, _, _) = post(&addr, "/v1/messages", &follow);
    assert_eq!(status, 200);
    let second = ctx.text.decode(&engine.last().0).unwrap();
    assert!(second.starts_with(&format!("{first_text}I should list files.</think>")), "{second}");
    assert!(second.contains("a.txt"));

    // count_tokens is the rendered prompt's length; too long is Anthropic's wording
    let (_, _, body) = post(&addr, "/v1/messages/count_tokens", &req);
    let (msgs, tools) = to_chat(&req, None, false).unwrap();
    let want = ctx.text.encode(&render(&msgs, Some(&Value::Array(tools)), Some("low")).unwrap()).unwrap().len();
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["input_tokens"].as_u64(), Some(want as u64));
    let mut big = req.clone();
    big["messages"] = json!([{"role": "user", "content": "word ".repeat(70000)}]);
    let (status, _, body) = post(&addr, "/v1/messages", &big);
    let err: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((status, err["type"].as_str()), (400, Some("error")));
    assert!(err["error"]["message"].as_str().unwrap().starts_with("prompt is too long:"), "{err}");
    let (status, _, body) = post(&addr, "/v1/messages/batches", &req);
    assert_eq!((status, serde_json::from_str::<Value>(&body).unwrap()["error"]["type"].as_str()), (404, Some("not_found_error")));
}

#[test]
fn glm_openai_routes() {
    let Some((addr, engine, ctx)) = glm() else {
        eprintln!("GLM tokenizer not present; skipping");
        return;
    };
    let set = |s: &str| *engine.answer.lock().unwrap() = ctx.text.encode(s).unwrap();
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
                        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
    let hist = json!([{"role": "user", "content": "Weather?"},
                      {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]);
    // the round trip: a rendered call parses back into the same arguments (test_glm_app.py)
    let rendered = render(hist.as_array().unwrap(), Some(&tools), None).unwrap();
    let call = &rendered[rendered.rfind("<tool_call>").unwrap()..rendered.rfind("</tool_call>").unwrap() + "</tool_call>".len()];
    // drafts accepted past the stop id ("never") are not the answer
    set(&format!("ok</think>Checking.{call}<|observation|>never seen"));
    let (status, _, body) = post(&addr, "/v1/chat/completions", &json!({"messages": hist, "tools": tools, "max_tokens": 100000}));
    assert_eq!(status, 200, "{body}");
    let r: Value = serde_json::from_str(&body).unwrap();
    let (prompt, max_tokens, stop) = engine.last();
    assert_eq!(ctx.text.decode(&prompt).unwrap(), render(hist.as_array().unwrap(), Some(&tools), Some("low")).unwrap());
    // clamped to the context: max_seq - prompt - 1
    assert_eq!((max_tokens, stop), (65536 - prompt.len() - 1, vec![154820, 154827, 154829]));
    let m = &r["choices"][0]["message"];
    assert_eq!((m["content"].as_str(), m["reasoning_content"].as_str()), (Some("Checking."), Some("ok")));
    assert_eq!(m["tool_calls"][0]["function"], json!({"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}));
    assert!(m["tool_calls"][0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
    let n = ctx.text.encode(&format!("ok</think>Checking.{call}<|observation|>")).unwrap().len();
    assert_eq!((r["usage"]["completion_tokens"].as_u64(), r["usage"]["muse"]["reused_prefix"].as_u64()), (Some(n as u64), Some(7)));
    assert_eq!(r["usage"]["total_tokens"].as_u64(), Some((n + prompt.len()) as u64));

    // a plain answer: content and reasoning split at </think>, the stop id dropped
    set("Let me think.</think>Héllo wörld!<|user|>");
    let (_, _, body) = post(&addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "Hi"}]}));
    let r: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(r["choices"][0]["message"], json!({"role": "assistant", "content": "Héllo wörld!", "reasoning_content": "Let me think."}));
    assert_eq!((r["choices"][0]["finish_reason"].as_str(), r["object"].as_str()), (Some("stop"), Some("chat.completion")));
    // max_tokens before </think>: length, no content
    let (_, _, body) = post(&addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2}));
    let r: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((r["choices"][0]["finish_reason"].as_str(), &r["choices"][0]["message"]["content"]), (Some("length"), &Value::Null));
    assert_eq!(engine.last().1, 2);
    // max_completion_tokens when max_tokens is absent or falsy
    post(&addr, "/v1/chat/completions", &json!({"messages": [], "max_tokens": 0, "max_completion_tokens": 5}));
    assert_eq!(engine.last().1, 5);

    // streamed: role, one chunk per routed delta, the calls in one chunk, usage at the end
    set(&format!("ok</think>Checking.{call}<|observation|>"));
    let (status, ctype, body) = post(&addr, "/v1/chat/completions", &json!({"messages": hist, "tools": tools, "stream": true}));
    assert_eq!((status, ctype.as_str()), (200, "text/event-stream"));
    assert!(body.ends_with("data: [DONE]\n\n"));
    let chunks = sse_data(&body);
    assert_eq!(chunks[0]["choices"][0]["delta"], json!({"role": "assistant"}));
    let joined = |k: &str| -> String { chunks.iter().filter_map(|c| c["choices"][0]["delta"][k].as_str()).collect() };
    assert_eq!((joined("reasoning_content"), joined("content")), ("ok".to_string(), "Checking.".to_string()));
    let calls = chunks.iter().find(|c| c["choices"][0]["delta"].get("tool_calls").is_some()).unwrap();
    let c0 = &calls["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!((c0["index"].as_u64(), &c0["function"]["name"]), (Some(0), &json!("get_weather")));
    let end = chunks.last().unwrap();
    assert_eq!((end["choices"][0]["finish_reason"].as_str(), end["choices"][0]["delta"].clone()), (Some("tool_calls"), json!({})));
    assert_eq!(end["usage"]["completion_tokens"].as_u64(), Some(n as u64));
    assert_eq!(end["x_engine_stats"]["reused_prefix"].as_u64(), Some(7));

    // the effort: request fields first, else the server's
    let effort = |body: Value| -> String {
        let (_, _, b) = post(&addr, "/v1/debug/prompt", &body);
        let v: Value = serde_json::from_str(&b).unwrap();
        let p = v["prompt"].as_str().unwrap();
        assert_eq!(v["prompt_tokens"].as_u64(), Some(v["ids"].as_array().unwrap().len() as u64));
        p["[gMASK]<sop><|system|>Reasoning Effort: ".len()..].split('<').next().unwrap().to_string()
    };
    let msgs = json!([{"role": "user", "content": "Hi"}]);
    assert_eq!(effort(json!({"messages": msgs})), "Low");
    assert_eq!(effort(json!({"messages": msgs, "chat_template_kwargs": {"reasoning_effort": "high"}})), "High");
    assert_eq!(effort(json!({"messages": msgs, "reasoning_effort": "max", "chat_template_kwargs": {"reasoning_effort": "high"}})), "Max");
    assert_eq!(effort(json!({"messages": msgs, "reasoning": {"effort": "high"}, "reasoning_effort": "low"})), "High");
    assert_eq!(effort(json!({"messages": msgs, "reasoning_effort": 3})), "Max");

    // too long; not JSON; a message the template refuses
    let (status, _, body) = post(&addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "word ".repeat(70000)}]}));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("exceeds the context"));
    let (status, _, _) = post(&addr, "/v1/chat/completions", &json!("not an object"));
    assert_eq!(status, 400);
    let (status, _, _) = post(&addr, "/v1/chat/completions", &json!({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}]}));
    assert_eq!(status, 400);

    // /v1/completions: a raw prompt, the text is every answer id through the stop id
    set("A reply.<|endoftext|>more");
    let (_, _, body) = post(&addr, "/v1/completions", &json!({"prompt": "Once upon a time"}));
    let r: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(r["choices"][0], json!({"index": 0, "text": "A reply.<|endoftext|>", "finish_reason": "stop"}));
    assert_eq!(engine.last().0, ctx.text.encode("Once upon a time").unwrap());
}

#[test]
fn request_effort_precedence() {
    // server/test_muse_app.py's RequestEffortTests
    let f = |v: Value| request_effort(&v).cloned();
    let ctk = json!({"reasoning_effort": "high"});
    assert_eq!(f(json!({})), None);
    assert_eq!(f(json!({"chat_template_kwargs": ctk})), Some(json!("high")));
    assert_eq!(f(json!({"chat_template_kwargs": ctk, "reasoning_effort": "max"})), Some(json!("max")));
    assert_eq!(f(json!({"chat_template_kwargs": ctk, "reasoning_effort": "max", "reasoning": {"effort": "low"}})), Some(json!("low")));
    assert_eq!(f(json!({"reasoning_effort": null, "chat_template_kwargs": {"reasoning_effort": ""}})), None);
    assert_eq!(f(json!({"reasoning": "high", "chat_template_kwargs": null})), None);
}

#[test]
fn ds_messages_route() {
    let Some(d) = dir("TIDELM_MODEL_DIR", "models/DeepSeek-V4.1-Flash-1017") else {
        eprintln!("DeepSeek tokenizer not present; skipping");
        return;
    };
    let text = Text::load(&d).unwrap();
    let eos = text.encode(tidelm_text::chat::EOS).unwrap()[0];
    let (addr, engine, ctx) = serve(text, Family::Ds, eos, 65536);
    let answer = "Let me look.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"Bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">ls -la\
                  </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
    *engine.answer.lock().unwrap() = ctx.text.encode(answer).unwrap();
    let req = anthropic_req();
    let (status, _, body) = post(&addr, "/v1/messages", &req);
    assert_eq!(status, 200, "{body}");
    let msg: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(msg["content"][0], json!({"type": "text", "text": "Let me look."}));
    assert_eq!((msg["content"][1]["name"].as_str(), &msg["content"][1]["input"]), (Some("Bash"), &json!({"command": "ls -la"})));
    assert_eq!((msg["stop_reason"].as_str(), msg["usage"]["cache_read_input_tokens"].as_u64()), (Some("tool_use"), Some(7)));
    let (prompt, max_tokens, stop) = engine.last();
    assert_eq!((max_tokens, stop), (512, vec![eos]));
    let p = ctx.text.decode(&prompt).unwrap();
    assert!(p.starts_with(tidelm_text::chat::BOS) && p.contains("You are an agent.") && p.contains("Bash") && !p.contains("billing-header"), "{p}");
    let mut streamed = req.clone();
    streamed["stream"] = json!(true);
    let (_, _, body) = post(&addr, "/v1/messages", &streamed);
    let (kinds, events) = kinds_and_events(&body);
    assert_eq!(kinds, ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start",
                       "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
    let text: String = events.iter().filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
        .map(|e| e["delta"]["text"].as_str().unwrap()).collect();
    assert_eq!(text, "Let me look.");
    let (_, _, body) = post(&addr, "/v1/messages/count_tokens", &req);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["input_tokens"].as_u64(), Some(prompt.len() as u64));
}

/// A response with what differs by run or by engine taken out: ids, timestamps, the engine's
/// stats; SSE keepalives and pings (timing) dropped.
fn normalize(ctype: &str, body: &str) -> Value {
    fn norm(v: &mut Value) {
        match v {
            Value::Object(m) => {
                for (k, x) in m.iter_mut() {
                    match (k.as_str(), &*x) {
                        ("id", Value::String(s)) => {
                            if let Some(p) = ["chatcmpl-", "call_", "toolu_", "msg_"].iter().find(|p| s.starts_with(*p)) {
                                *x = json!(format!("{p}*"));
                            }
                        }
                        ("created", _) => *x = json!(0),
                        ("muse" | "x_engine_stats", _) => *x = json!({}),
                        _ => norm(x),
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(norm),
            _ => {}
        }
    }
    if ctype != "text/event-stream" {
        let mut v: Value = serde_json::from_str(body).unwrap();
        norm(&mut v);
        return v;
    }
    body.split("\n\n")
        .filter(|c| !c.is_empty() && !c.starts_with(':'))
        .filter_map(|c| {
            let (event, data) = match c.split_once('\n') {
                Some((e, d)) if e.starts_with("event: ") => (Some(e), d),
                _ => (None, c),
            };
            let data = data.strip_prefix("data: ").unwrap();
            if data == "[DONE]" {
                return Some(json!("[DONE]"));
            }
            let mut v: Value = serde_json::from_str(data).unwrap();
            if v["type"] == "ping" {
                return None;
            }
            norm(&mut v);
            Some(json!([event, v]))
        })
        .collect()
}

#[test]
fn glm_routes_answer_as_muse_app() {
    let Some((addr, engine, ctx)) = glm() else {
        eprintln!("GLM tokenizer not present; skipping");
        return;
    };
    let fixture: Value = serde_json::from_str(include_str!("../tests/glm_server_cases.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 20);
    let mut prev = String::new();
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let body = if c.get("follow").is_some() {
            // the follow-up of the previous reply, built from this server's reply (its ids)
            let msg: Value = serde_json::from_str(&prev).unwrap();
            let id = msg["content"].as_array().unwrap().iter().find(|b| b["type"] == "tool_use").unwrap()["id"].clone();
            let mut req = anthropic_req_of(cases);
            req["messages"].as_array_mut().unwrap().extend([
                json!({"role": "assistant", "content": msg["content"]}),
                json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": "a.txt"}]})]);
            req
        } else if let Some(n) = c["big"].as_u64() {
            let mut req = anthropic_req_of(cases);
            req["messages"] = json!([{"role": "user", "content": "word ".repeat(n as usize)}]);
            req
        } else {
            c["body"].clone()
        };
        *engine.answer.lock().unwrap() = ctx.text.encode(c["answer"].as_str().unwrap()).unwrap();
        let jobs = engine.jobs.lock().unwrap().len();
        let (status, ctype, resp) = post(&addr, c["path"].as_str().unwrap(), &body);
        assert_eq!(status as u64, c["status"].as_u64().unwrap(), "{name}: {resp}");
        let want_ctype = c["content_type"].as_str().unwrap();
        assert_eq!(ctype, want_ctype, "{name}");
        let mut got = normalize(&ctype, &resp);
        let want = normalize(want_ctype, c["response"].as_str().unwrap());
        if c["path"] == "/v1/debug/prompt" {
            // this server also returns the ids and the effort
            got.as_object_mut().unwrap().retain(|k, _| want.get(k).is_some());
        }
        assert_eq!(got, want, "{name}");
        match c.get("prompt") {
            Some(p) => {
                let (prompt, max_tokens, _) = engine.last();
                assert_eq!(engine.jobs.lock().unwrap().len(), jobs + 1, "{name}: one job");
                assert_eq!(Value::from(prompt), *p, "{name}: the engine's prompt");
                assert_eq!(max_tokens as u64, c["max_tokens"].as_u64().unwrap(), "{name}: max_tokens");
            }
            None => assert_eq!(engine.jobs.lock().unwrap().len(), jobs, "{name}: no job"),
        }
        prev = resp;
    }
}

/// the fixture's first /v1/messages body: the request a follow-up extends
fn anthropic_req_of(cases: &[Value]) -> Value {
    cases.iter().find(|c| c["path"] == "/v1/messages" && c.get("body").is_some()).unwrap()["body"].clone()
}
