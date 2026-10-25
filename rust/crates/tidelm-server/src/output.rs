//! An answer's text on its way out (server/app.py's OutputRouter and _parse_tool_calls):
//! reasoning until `</think>`, then content until a tool-calls block opens, and that block,
//! which is held back whole and parsed once the answer is done. The request's stop strings
//! end reasoning or content (not a tool block). Text that may still turn out to begin a marker
//! is held until the next tokens decide it, so the "\n\n" before a block never reaches content.
//!
//! [`Answer`] is either family's router behind one face (GLM's routes and /v1/messages use
//! it): DeepSeek's text router over the detokenized stream, or GLM's id router
//! (tidelm_text::glm::GlmParser, server/glm_app.py).

use serde_json::{json, Value};
use tidelm_text::chat::{parse_completion, parse_tool_calls_tolerant, EOS, THINK_END, TOOL_CALLS_START};
use tidelm_text::glm::{Channel, GlmIds, GlmParser};
use tidelm_text::Text;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Reasoning,
    Content,
    Tool,
}

pub struct OutputRouter {
    pub phase: Phase,
    pending: String,
    stops: Vec<String>,
    pub reasoning: String,
    pub content: String,
    pub tool_text: String,
    /// a stop string ended the answer
    pub stopped: bool,
}

/// The longest suffix of `text` that is a proper prefix of a marker, in bytes.
fn held_suffix_len(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        // proper prefixes of m, longest first, at its character boundaries
        let ends: Vec<usize> = m.char_indices().map(|(i, _)| i).skip(1).filter(|&k| k <= text.len()).collect();
        if let Some(&k) = ends.iter().rev().find(|&&k| text.ends_with(&m[..k])) {
            best = best.max(k);
        }
    }
    best
}

impl OutputRouter {
    pub fn new(thinking: bool, stops: Vec<String>) -> Self {
        OutputRouter {
            phase: if thinking { Phase::Reasoning } else { Phase::Content },
            pending: String::new(),
            stops,
            reasoning: String::new(),
            content: String::new(),
            tool_text: String::new(),
            stopped: false,
        }
    }

    fn markers(&self) -> Vec<&str> {
        let first = match self.phase {
            Phase::Reasoning => THINK_END,
            Phase::Content => TOOL_CALLS_START,
            Phase::Tool => return Vec::new(),
        };
        std::iter::once(first).chain(self.stops.iter().map(|s| s.as_str())).collect()
    }

    fn emit(&mut self, text: &str, out: &mut Vec<(Phase, String)>) {
        if text.is_empty() {
            return;
        }
        match self.phase {
            Phase::Reasoning => self.reasoning.push_str(text),
            _ => self.content.push_str(text),
        }
        out.push((self.phase, text.to_string()));
    }

    /// The (phase, text) pieces that are now certain.
    pub fn feed(&mut self, text: &str) -> Vec<(Phase, String)> {
        let mut out = Vec::new();
        self.pending.push_str(text);
        while !self.pending.is_empty() && !self.stopped {
            if self.phase == Phase::Tool {
                self.tool_text.push_str(&std::mem::take(&mut self.pending));
                break;
            }
            let markers: Vec<String> = self.markers().into_iter().map(String::from).collect();
            let refs: Vec<&str> = markers.iter().map(|s| s.as_str()).collect();
            // the earliest marker; at one place the first listed
            let hit = refs
                .iter()
                .enumerate()
                .filter_map(|(i, m)| self.pending.find(m).map(|p| (p, i)))
                .min()
                .map(|(p, i)| (p, refs[i]));
            let pending = std::mem::take(&mut self.pending);
            let Some((pos, m)) = hit else {
                let keep = held_suffix_len(&pending, &refs);
                self.emit(&pending[..pending.len() - keep], &mut out);
                self.pending = pending[pending.len() - keep..].to_string();
                break;
            };
            self.emit(&pending[..pos], &mut out);
            if m == THINK_END && self.phase == Phase::Reasoning {
                self.phase = Phase::Content;
                self.pending = pending[pos + m.len()..].to_string();
            } else if m == TOOL_CALLS_START && self.phase == Phase::Content {
                self.phase = Phase::Tool;
                self.pending = pending[pos..].to_string();
            } else {
                self.stopped = true;
            }
        }
        out
    }

    /// The answer is over: what was held back.
    pub fn finish(&mut self) -> Vec<(Phase, String)> {
        let mut out = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        if self.phase == Phase::Tool {
            self.tool_text.push_str(&pending);
        } else if !self.stopped {
            self.emit(&pending, &mut out);
        }
        out
    }

    /// The tool calls in the answer, OpenAI-shaped (`call_` ids): the checkpoint's parser,
    /// else the tolerant one; when neither finds one the block's text goes back into content
    /// (returned, so a stream can send what it never saw).
    pub fn tool_calls(&mut self, thinking: bool) -> (Vec<Value>, String) {
        if self.tool_text.is_empty() {
            return (Vec::new(), String::new());
        }
        let mut text = String::new();
        if thinking {
            text.push_str(&self.reasoning);
            text.push_str(THINK_END);
        }
        text.push_str(&self.content);
        text.push_str(&self.tool_text);
        if !text.ends_with(EOS) {
            text.push_str(EOS);
        }
        let calls: Vec<(String, String)> = match parse_completion(&text, thinking) {
            Ok(p) => p.tool_calls.iter().map(|c| (c.full_name(), c.arguments.clone())).collect(),
            Err(e) => {
                let recovered = parse_tool_calls_tolerant(&text);
                if recovered.is_empty() {
                    eprintln!("tool-call parse failed ({e}); returning raw text as content");
                    let raw = std::mem::take(&mut self.tool_text);
                    self.content.push_str(&raw);
                    return (Vec::new(), raw);
                }
                eprintln!("tool-call strict parse failed ({e}); recovered {} call(s) tolerantly", recovered.len());
                recovered
            }
        };
        let calls = calls
            .into_iter()
            .map(|(name, arguments)| json!({"id": format!("call_{}", hex24()), "type": "function",
                                            "function": {"name": name, "arguments": arguments}}))
            .collect();
        (calls, String::new())
    }
}

/// Incremental detokenization (the usual prefix/read offsets): text is released once it no
/// longer ends in a partial UTF-8 character.
#[derive(Default)]
pub struct Detok {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    pub fn push(&mut self, text: &Text, t: u32) -> String {
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
    pub fn flush(&mut self, text: &Text) -> String {
        let full = text.decode(&self.ids[self.prefix..]).unwrap_or_default();
        let prev = text.decode(&self.ids[self.prefix..self.read]).unwrap_or_default();
        (self.prefix, self.read) = (self.ids.len(), self.ids.len());
        if full.len() > prev.len() { full.get(prev.len()..).unwrap_or("").to_string() } else { String::new() }
    }
}

/// One answer's tokens on their way to a client, whichever the family: the visible pieces as
/// they become certain, then the tool calls.
pub trait Answer: Send {
    /// One step's tokens: the (phase, text) pieces now certain. Tokens past the answer's end
    /// (a stop token or string; a step can emit accepted drafts after it) are ignored.
    fn feed(&mut self, text: &Text, toks: &[u32]) -> Vec<(Phase, String)>;
    /// The answer is over: what was held back.
    fn finish(&mut self, text: &Text) -> Vec<(Phase, String)>;
    /// a stop token or stop string ended it: the engine's later tokens are not the answer's
    fn ended(&self) -> bool;
    /// its tokens, through the stop token
    fn tokens(&self) -> usize;
    fn reasoning(&self) -> &str;
    fn content(&self) -> &str;
    /// The tool calls (OpenAI-shaped, `call_` ids), and the text of a block no parser could
    /// read, which went back into content (a stream has not sent it).
    fn tool_calls(&mut self, text: &Text, tools: Option<&Value>) -> (Vec<Value>, String);
}

/// GLM: server/glm_app.py's GlmParser on the ids, the stream cut after the first stop id (the
/// Python engine yields nothing past it).
pub struct GlmAnswer {
    parser: GlmParser,
    stop: Vec<u32>,
    ended: bool,
    n: usize,
}

impl GlmAnswer {
    pub fn new(ids: &GlmIds) -> Self {
        GlmAnswer { parser: GlmParser::new(ids), stop: ids.stop.clone(), ended: false, n: 0 }
    }
}

fn phase(c: Channel) -> Phase {
    match c {
        Channel::Reasoning => Phase::Reasoning,
        Channel::Content => Phase::Content,
    }
}

impl Answer for GlmAnswer {
    fn feed(&mut self, text: &Text, toks: &[u32]) -> Vec<(Phase, String)> {
        let mut out = Vec::new();
        for &t in toks {
            if self.ended {
                break;
            }
            self.n += 1;
            out.extend(self.parser.feed(text, t).into_iter().map(|(c, s)| (phase(c), s)));
            self.ended = self.stop.contains(&t);
        }
        out
    }

    fn finish(&mut self, text: &Text) -> Vec<(Phase, String)> {
        self.parser.finish(text).into_iter().map(|(c, s)| (phase(c), s)).collect()
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn tokens(&self) -> usize {
        self.n
    }

    fn reasoning(&self) -> &str {
        &self.parser.reasoning
    }

    fn content(&self) -> &str {
        &self.parser.content
    }

    fn tool_calls(&mut self, text: &Text, tools: Option<&Value>) -> (Vec<Value>, String) {
        let calls = self.parser.tool_calls(text, tools).into_iter()
            .map(|(name, arguments)| json!({"id": format!("call_{}", hex24()), "type": "function",
                                            "function": {"name": name, "arguments": arguments}}))
            .collect();
        (calls, String::new())
    }
}

/// DeepSeek: the detokenized stream through OutputRouter, ending at the end-of-sentence id or
/// a stop string (api.rs's chat loop).
pub struct DsAnswer {
    router: OutputRouter,
    detok: Detok,
    eos: u32,
    thinking: bool,
    hit_eos: bool,
    n: usize,
}

impl DsAnswer {
    pub fn new(thinking: bool, stops: Vec<String>, eos: u32) -> Self {
        DsAnswer { router: OutputRouter::new(thinking, stops), detok: Detok::default(), eos, thinking, hit_eos: false, n: 0 }
    }
}

impl Answer for DsAnswer {
    fn feed(&mut self, text: &Text, toks: &[u32]) -> Vec<(Phase, String)> {
        let mut out = Vec::new();
        for &t in toks {
            if self.hit_eos || self.router.stopped {
                break;
            }
            self.n += 1;
            if t == self.eos {
                self.hit_eos = true;
                continue;
            }
            out.extend(self.router.feed(&self.detok.push(text, t)));
        }
        out
    }

    fn finish(&mut self, text: &Text) -> Vec<(Phase, String)> {
        let mut tail = self.router.feed(&self.detok.flush(text));
        tail.extend(self.router.finish());
        tail
    }

    fn ended(&self) -> bool {
        self.hit_eos || self.router.stopped
    }

    fn tokens(&self) -> usize {
        self.n
    }

    fn reasoning(&self) -> &str {
        &self.router.reasoning
    }

    fn content(&self) -> &str {
        &self.router.content
    }

    fn tool_calls(&mut self, _text: &Text, _tools: Option<&Value>) -> (Vec<Value>, String) {
        self.router.tool_calls(self.thinking)
    }
}

/// 24 random hex digits (uuid4().hex[:24])
pub fn hex24() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let word = || {
        let mut h = RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        h.finish()
    };
    format!("{:016x}{:016x}", word(), word())[..24].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(r: &mut OutputRouter, pieces: &[&str]) -> Vec<(Phase, String)> {
        let mut out: Vec<(Phase, String)> = pieces.iter().flat_map(|p| r.feed(p)).collect();
        out.extend(r.finish());
        out
    }

    #[test]
    fn the_newlines_before_a_block_never_reach_content() {
        let mut r = OutputRouter::new(false, vec![]);
        let call = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"true\">1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let out = run(&mut r, &["Let me look", " that up.\n", "\n", call]);
        assert_eq!(out, vec![(Phase::Content, "Let me look".into()), (Phase::Content, " that up.".into())]);
        assert_eq!(r.content, "Let me look that up.");
        assert_eq!(r.tool_text, format!("\n\n{call}"));
        let (calls, raw) = r.tool_calls(false);
        assert!(raw.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"], json!({"name": "f", "arguments": "{\"x\": \"1\"}"}));
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_") && calls[0]["id"].as_str().unwrap().len() == 29);
    }

    #[test]
    fn reasoning_ends_at_think_end_and_stops_cut_both_phases() {
        let mut r = OutputRouter::new(true, vec!["STOP".into()]);
        let out = run(&mut r, &["thinking </th", "ink>answer ST", "OP never"]);
        assert_eq!(out, vec![(Phase::Reasoning, "thinking ".into()), (Phase::Content, "answer ".into())]);
        assert!(r.stopped);
        let mut r = OutputRouter::new(true, vec!["halt".into()]);
        run(&mut r, &["a hal", "t b"]);
        assert_eq!((r.reasoning.as_str(), r.content.as_str(), r.stopped), ("a ", "", true));
    }

    #[test]
    fn a_held_prefix_is_released_when_it_turns_out_not_to_be_one() {
        let mut r = OutputRouter::new(false, vec![]);
        assert_eq!(r.feed("x\n\n<｜DS"), vec![(Phase::Content, "x".into())]);
        assert_eq!(r.feed("ok"), vec![(Phase::Content, "\n\n<｜DSok".into())]);
        // a multi-byte character split across the held part
        assert_eq!(held_suffix_len("ab\n\n<｜", &[TOOL_CALLS_START]), "\n\n<｜".len());
    }

    #[test]
    fn an_unparsable_block_returns_to_content() {
        let mut r = OutputRouter::new(false, vec![]);
        run(&mut r, &["hi\n\n<｜DSML｜ calls>\nnonsense"]);
        let (calls, raw) = r.tool_calls(false);
        assert!(calls.is_empty());
        assert_eq!(raw, "\n\n<｜DSML｜ calls>\nnonsense");
        assert_eq!(r.content, "hi\n\n<｜DSML｜ calls>\nnonsense");
    }
}
