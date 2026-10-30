//! Minimal HTTP/1.1: one request a connection (Connection: close), bodies by Content-Length,
//! JSON responses and server-sent event streams.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use crate::{anthropic, api, Ctx};

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    /// whether a key was sent: `Authorization: Bearer <key>` (OpenAI's clients, and Anthropic's
    /// with an auth token) or `x-api-key: <key>` (Anthropic's with an API key)
    pub sent_key: bool,
    /// the server wants a key and none sent matched: the body was not read
    pub refused: bool,
}

/// A client-visible error in OpenAI's shape.
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<&'static str>,
}

impl ApiError {
    pub fn bad(message: impl Into<String>, param: Option<&'static str>) -> Self {
        ApiError { status: 400, message: message.into(), param }
    }
}

/// The routes answered without a key: the health check.
fn open_route(method: &str, path: &str) -> bool {
    matches!((method, path), ("GET", "/health" | ""))
}

/// One request; with `api_keys`, one that carries none of them is refused before its body is read.
fn read_request(stream: &TcpStream, api_keys: &[String]) -> std::io::Result<Option<Request>> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    // the query goes; a trailing "/" goes too (Claude Code posts /v1/messages?beta=true)
    let path = target.split('?').next().unwrap_or("").trim_end_matches('/').to_string();
    let (mut len, mut keys) = (0usize, vec![]);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let (k, v) = (k.trim(), v.trim());
            if k.eq_ignore_ascii_case("content-length") {
                len = v.parse().unwrap_or(0);
            } else if k.eq_ignore_ascii_case("x-api-key") {
                keys.push(v.to_string());
            } else if k.eq_ignore_ascii_case("authorization") {
                if let Some((scheme, token)) = v.split_once(' ') {
                    if scheme.eq_ignore_ascii_case("bearer") {
                        keys.push(token.trim().to_string());
                    }
                }
            }
        }
    }
    let refused = !api_keys.is_empty() && !open_route(&method, &path) && !key_ok(api_keys, &keys);
    let mut body = vec![0u8; if refused { 0 } else { len }];
    r.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body, sent_key: !keys.is_empty(), refused }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub fn send_json(stream: &mut TcpStream, status: u16, v: &serde_json::Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(v).unwrap_or_default();
    write!(stream, "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
           reason(status), body.len())?;
    stream.write_all(&body)?;
    stream.flush()
}

pub fn send_error(stream: &mut TcpStream, e: &ApiError) -> std::io::Result<()> {
    let kind = if e.status >= 500 { "server_error" } else { "invalid_request_error" };
    send_json(stream, e.status, &serde_json::json!({
        "error": {"message": e.message, "type": kind, "param": e.param, "code": null}
    }))
}

pub fn start_events(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    stream.flush()
}

pub fn send_event(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(stream, "data: {data}\n\n")?;
    stream.flush()
}

/// Whether a key the client sent is one of the server's, compared in time independent of where
/// they differ.
fn key_ok(server: &[String], sent: &[String]) -> bool {
    sent.iter().flat_map(|s| server.iter().map(move |k| (s, k))).fold(false, |ok, (s, k)| {
        ok | (k.len() == s.len() && k.bytes().zip(s.bytes()).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0)
    })
}

/// One connection: read its request, route it, answer, close.
pub fn serve(mut stream: TcpStream, ctx: &Ctx) {
    let _ = stream.set_nodelay(true);
    let req = match read_request(&stream, &ctx.api_keys) {
        Ok(Some(r)) => r,
        _ => return,
    };
    let path = req.path.as_str();
    // with keys set (TIDELM_API_KEY), every route but the health check wants one of them, refused in
    // the route's own error shape
    if req.refused {
        let message = if req.sent_key { "invalid API key" } else { "an API key is required: Authorization: Bearer <key> or x-api-key: <key>" };
        let body = if path.starts_with("/v1/messages") {
            serde_json::json!({"type": "error", "error": {"type": "authentication_error", "message": message}})
        } else {
            serde_json::json!({"error": {"message": message, "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}})
        };
        let _ = send_json(&mut stream, 401, &body);
        return;
    }
    let res = match (req.method.as_str(), path) {
        (m, p) if open_route(m, p) => send_json(&mut stream, 200, &serde_json::json!({"status": "ok", "engine": "tidelm-rust"})).map_err(Into::into),
        ("GET", "/v1/models") => send_json(&mut stream, 200, &serde_json::json!({
            "object": "list",
            "data": [{"id": ctx.model, "object": "model", "owned_by": "tidelm"}],
        })).map_err(Into::into),
        ("POST", p) if p.starts_with("/v1/messages") => anthropic::serve(&mut stream, p, &req.body, ctx),
        ("POST", "/v1/chat/completions") => api::chat(&mut stream, &req.body, ctx),
        ("POST", "/v1/debug/prompt") => api::debug_prompt(&mut stream, &req.body, ctx),
        ("POST", "/v1/completions") => match &ctx.family {
            crate::Family::Glm(g) => crate::glm::completions(&mut stream, &req.body, ctx, g),
            crate::Family::Ds => Err(api::Failure::Api(ApiError { status: 404, message: "no route for POST /v1/completions".into(), param: None })),
        },
        (m, p) => Err(api::Failure::Api(ApiError { status: 404, message: format!("no route for {m} {p}"), param: None })),
    };
    match res {
        Ok(()) => {}
        Err(api::Failure::Api(e)) => {
            let _ = send_error(&mut stream, &e);
        }
        Err(api::Failure::Io) => {}
    }
}
