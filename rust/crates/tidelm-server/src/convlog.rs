//! Conversation log for dreaming: server/conversation_log.py's records, from the Rust server.
//!
//! Opt-in with DSV41_DREAM=1 (either engine); off, nothing is written. Files are per day under
//! DSV41_DREAM_DIR (default /models/dream)/conversations/YYYY-MM-DD.jsonl, appended by rank 0's HTTP
//! threads (the peer never sees requests); dream/extract_facts.py reads them as it reads Python's.
//!
//! Each record: {"ts", "model", "family", "thinking", "effort", "messages": [request messages],
//! "reply": {"reasoning", "content", "tool_calls"}, "finish_reason", "prompt_ids", "token_ids"}.
//! The ids are the exact sequence the engine ran (a drafter's training text: prompt_ids then
//! token_ids). Ids cannot be redacted, so a record whose text had anything scrubbed keeps its
//! scrubbed text and null ids: a dream loses that record's ids, the disk never holds the secret.
//!
//! Never written, as in Python:
//!   * `@@` command turns and their replies (the Python server's commands; a history may still
//!     carry them) -- a dream must not learn the wake word or the dreaming notice;
//!   * requests whose OpenAI `user` is "tidelm-dream" -- a dream's own extraction and evals;
//!   * anything that looks like a secret (scrub): a secret trained into weights can come out in any
//!     later answer and cannot be removed selectively. The patterns are deliberately broad: a false
//!     positive costs one redacted word, a miss costs a secret on disk.

use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use regex::{Captures, Regex};
use serde_json::{json, Value};

/// The dream's own requests carry this OpenAI `user` and are never logged
pub const DREAM_USER: &str = "tidelm-dream";
/// server/commands.py's MARKER: prefixed to every command reply
const MARKER: char = '\u{2063}';

/// Where the records go; built once at boot (Log::from_env), held by Ctx
pub struct Log {
    dir: PathBuf,
    lock: Mutex<()>,
}

/// One finished completion, as the route that served it has it
pub struct Turn<'a> {
    /// the request body (its `messages` and `user`)
    pub body: &'a Value,
    pub model: &'a str,
    pub family: &'static str,
    pub thinking: bool,
    pub effort: Value,
    pub reasoning: &'a str,
    pub content: &'a str,
    pub tool_calls: &'a [Value],
    pub finish_reason: &'a str,
    pub prompt_ids: &'a [u32],
    pub token_ids: &'a [u32],
}

impl Log {
    /// DSV41_DREAM=1: a log under DSV41_DREAM_DIR; anything else: None (nothing is written)
    pub fn from_env() -> Option<Log> {
        if std::env::var("DSV41_DREAM").as_deref() != Ok("1") {
            return None;
        }
        let dir = std::env::var("DSV41_DREAM_DIR").unwrap_or_else(|_| "/models/dream".into());
        eprintln!("conversation log: on, {dir}/conversations (DSV41_DREAM=1)");
        Some(Log::new(PathBuf::from(dir)))
    }

    pub fn new(dream_dir: PathBuf) -> Log {
        Log { dir: dream_dir.join("conversations"), lock: Mutex::new(()) }
    }

    /// Append one record. Never fails the request: an error is logged and dropped.
    pub fn record(&self, t: &Turn) {
        if t.body.get("user").and_then(Value::as_str) == Some(DREAM_USER) {
            return;
        }
        let msgs: Vec<&Value> = t.body.get("messages").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
        let msgs: Vec<Value> = msgs.into_iter().filter(|m| !is_command_turn(m)).cloned().collect();
        if msgs.is_empty() {
            return;
        }
        let mut hit = false;
        let messages = scrub_value(&Value::Array(msgs), &mut hit);
        let reply = json!({"reasoning": scrub(t.reasoning, &mut hit), "content": scrub(t.content, &mut hit),
                           "tool_calls": scrub_value(&Value::Array(t.tool_calls.to_vec()), &mut hit)});
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let row = json!({
            "ts": ts, "model": t.model, "family": t.family, "thinking": t.thinking, "effort": t.effort,
            "messages": messages, "reply": reply, "finish_reason": t.finish_reason,
            "prompt_ids": if hit { Value::Null } else { json!(t.prompt_ids) },
            "token_ids": if hit { Value::Null } else { json!(t.token_ids) },
        });
        if let Err(e) = self.append(ts, &row) {
            eprintln!("conversation log: write failed ({e}); the request is unaffected");
        }
    }

    fn append(&self, ts: f64, row: &Value) -> std::io::Result<()> {
        let mut line = row.to_string();
        line.push('\n');
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&self.dir)?;
        let path = self.dir.join(format!("{}.jsonl", local_date(ts as i64)));
        let mut f = std::fs::OpenOptions::new().append(true).create(true).mode(0o600).open(path)?;
        f.write_all(line.as_bytes())
    }
}

/// YYYY-MM-DD in the server's local time, as Python's time.localtime names the files
fn local_date(secs: i64) -> String {
    let t: libc::time_t = secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
        return "unknown-date".into();
    }
    format!("{:04}-{:02}-{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday)
}

/// The text parts of a message's content (a string, or a list's text blocks)
fn text_parts(content: Option<&Value>) -> Vec<&str> {
    match content {
        Some(Value::String(s)) => vec![s.as_str()],
        Some(Value::Array(a)) => a.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect(),
        _ => vec![],
    }
}

/// A user `@@` command or a command reply (server/commands.py::is_command_turn)
fn is_command_turn(m: &Value) -> bool {
    static CMD: OnceLock<Regex> = OnceLock::new();
    let cmd = CMD.get_or_init(|| Regex::new(r"^@@[A-Za-z][\w-]*(?:[ \t]+\S+)*[ \t]*$").unwrap());
    let parts = text_parts(m.get("content"));
    match m.get("role").and_then(Value::as_str) {
        Some("assistant") => parts.iter().any(|p| p.starts_with(MARKER)),
        Some("user") => parts.iter().any(|p| cmd.is_match(p.trim())),
        _ => false,
    }
}

/// How a pattern's match is replaced: a fixed label, or a group kept around "[REDACTED]"
enum Kind {
    Label(&'static str),
    /// groups 1 and 2 kept, 3 redacted (bearer tokens, credential-named assignments)
    Keep12,
    /// groups 1 and 3 kept, 2 redacted (a URL's password)
    Url,
}

/// server/conversation_log.py's SECRET_PATTERNS, in its order
fn patterns() -> &'static [(Regex, Kind)] {
    static P: OnceLock<Vec<(Regex, Kind)>> = OnceLock::new();
    P.get_or_init(|| {
        let r = |s: &str| Regex::new(s).unwrap();
        vec![
            (r(r"\bhf_[A-Za-z0-9]{20,}\b"), Kind::Label("hf_")),
            (r(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"), Kind::Label("sk-")),
            (r(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), Kind::Label("gh_")),
            (r(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"), Kind::Label("github_pat_")),
            (r(r"\bAKIA[0-9A-Z]{16}\b"), Kind::Label("AKIA")),
            (r(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), Kind::Label("xox")),
            (r(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), Kind::Label("jwt")),
            (r(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"), Kind::Label("private key")),
            // before the assignment rule, which would otherwise take "Authorization: Bearer" and stop
            (r(r"(?i)\b(bearer)(\s+)([A-Za-z0-9._~+/=-]{16,})"), Kind::Keep12),
            (r(r#"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|AUTH)[A-Z0-9_]*)(\s*[:=]\s*["']?)([^\s"',;]{6,})"#),
             Kind::Keep12),
            (r(r"(?i)(://[^/\s:@]+:)([^/\s@]{3,})(@)"), Kind::Url),
        ]
    })
}

/// `text` with every secret pattern redacted; `hit` set when anything was
pub fn scrub(text: &str, hit: &mut bool) -> String {
    let mut out = text.to_string();
    for (re, kind) in patterns() {
        if !re.is_match(&out) {
            continue;
        }
        *hit = true;
        out = match kind {
            Kind::Label(l) => re.replace_all(&out, format!("[REDACTED {l}]").as_str()).into_owned(),
            Kind::Keep12 => re.replace_all(&out, |c: &Captures| format!("{}{}[REDACTED]", &c[1], &c[2])).into_owned(),
            Kind::Url => re.replace_all(&out, |c: &Captures| format!("{}[REDACTED]{}", &c[1], &c[3])).into_owned(),
        };
    }
    out
}

fn scrub_value(v: &Value, hit: &mut bool) -> Value {
    match v {
        Value::String(s) => Value::String(scrub(s, hit)),
        Value::Array(a) => Value::Array(a.iter().map(|x| scrub_value(x, hit)).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), scrub_value(x, hit))).collect()),
        x => x.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(s: &str) -> Value {
        json!({"role": "user", "content": s})
    }

    /// Records written by one Turn of `messages` (server/test_dream_commands.py's write)
    fn write(on: bool, body: Value, content: &str) -> Vec<Value> {
        let tmp = std::env::temp_dir().join(format!("convlog-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&tmp);
        if on {
            let log = Log::new(tmp.clone());
            log.record(&Turn { body: &body, model: "m", family: "glm", thinking: true, effort: json!("low"),
                               reasoning: "thinking", content, tool_calls: &[], finish_reason: "stop",
                               prompt_ids: &[1, 2, 3], token_ids: &[4, 5] });
        }
        let mut rows = vec![];
        if let Ok(rd) = std::fs::read_dir(tmp.join("conversations")) {
            for e in rd.flatten() {
                for line in std::fs::read_to_string(e.path()).unwrap().lines() {
                    rows.push(serde_json::from_str(line).unwrap());
                }
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
        rows
    }

    #[test]
    fn off_writes_nothing() {
        assert!(write(false, json!({"messages": [user("hello")]}), "x").is_empty());
    }

    #[test]
    fn command_turns_are_dropped() {
        let marked = format!("{MARKER}Last dream");
        let rows = write(true, json!({"messages": [user("@@dream log"), {"role": "assistant", "content": marked},
                                                   user("hello")]}), "x");
        let contents: Vec<&str> = rows[0]["messages"].as_array().unwrap().iter().map(|m| m["content"].as_str().unwrap()).collect();
        assert_eq!(contents, ["hello"]);
        assert_eq!(rows[0]["reply"]["reasoning"], "thinking");
        // nothing scrubbed: the exact ids ride along
        assert_eq!(rows[0]["prompt_ids"], json!([1, 2, 3]));
        assert_eq!(rows[0]["token_ids"], json!([4, 5]));
    }

    #[test]
    fn secrets_are_scrubbed_and_the_ids_dropped() {
        let secrets = [format!("hf_{}", "a1B2".repeat(9)), format!("sk-ant-{}", "x".repeat(30)), format!("ghp_{}", "Z".repeat(36)),
                       "AKIAABCDEFGHIJKLMNOP".into(), "HF_TOKEN=abcdefgh12345".into(), "password: hunter2hunter2".into(),
                       "https://bob:s3cretpw@host/x".into(), "Authorization: Bearer abcdefghijklmnopqrstuvwxyz".into()];
        let rows = write(true, json!({"messages": [user(&secrets.join("\n"))]}), &format!("token={}", "q".repeat(12)));
        let blob = rows[0].to_string();
        for leaked in ["a1B2a1B2", &"x".repeat(30), &"Z".repeat(36), "ABCDEFGHIJKLMNOP", "abcdefgh12345", "hunter2hunter2",
                       "s3cretpw", "abcdefghijklmnopqrstuvwxyz", &"q".repeat(12)] {
            assert!(!blob.contains(leaked), "{leaked} leaked");
        }
        assert!(blob.contains("HF_TOKEN=[REDACTED]"));
        assert!(rows[0]["prompt_ids"].is_null() && rows[0]["token_ids"].is_null());
    }

    #[test]
    fn dream_requests_are_not_logged() {
        assert!(write(true, json!({"messages": [user("extract facts")], "user": DREAM_USER}), "x").is_empty());
    }

    #[test]
    fn plain_text_is_untouched() {
        let text = "On Pocket Deck, ble_kb -r resets the BLE keyboard config. The token count was 512.";
        let rows = write(true, json!({"messages": [user(text)]}), "x");
        assert_eq!(rows[0]["messages"][0]["content"], text);
        assert_eq!(rows[0]["prompt_ids"], json!([1, 2, 3]));
    }
}
