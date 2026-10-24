//! Minimal HTTP/1.1: one request a connection (Connection: close), bodies by Content-Length,
//! JSON responses and server-sent event streams.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use crate::{api, Ctx};

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
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

fn read_request(stream: &TcpStream) -> std::io::Result<Option<Request>> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let path = target.split('?').next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
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

/// One connection: read its request, route it, answer, close.
pub fn serve(mut stream: TcpStream, ctx: &Ctx) {
    let _ = stream.set_nodelay(true);
    let req = match read_request(&stream) {
        Ok(Some(r)) => r,
        _ => return,
    };
    let res = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => send_json(&mut stream, 200, &serde_json::json!({"status": "ok", "engine": "tidelm-rust"})).map_err(Into::into),
        ("GET", "/v1/models") => send_json(&mut stream, 200, &serde_json::json!({
            "object": "list",
            "data": [{"id": ctx.model, "object": "model", "owned_by": "tidelm"}],
        })).map_err(Into::into),
        ("POST", "/v1/chat/completions") => api::chat(&mut stream, &req.body, ctx),
        ("POST", "/v1/debug/prompt") => api::debug_prompt(&mut stream, &req.body, ctx),
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
