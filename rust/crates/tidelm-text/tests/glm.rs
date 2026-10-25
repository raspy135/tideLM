//! tidelm_text::glm against server/glm_app.py's GlmTemplate (the checkpoint's
//! chat_template.jinja under jinja2) and GlmParser, on the cases tests/glm_cases.json records
//! (gen_glm_cases.py runs the Python): rendered text byte for byte, its ids from the Rust
//! tokenizer against the Python one's, and the parser's events, texts and tool calls on the
//! same ids. The tokenizer parts need the checkpoint (GLM_TOKENIZER_DIR, default
//! ~/models/GLM-5.3-EXL3-3.0bpw) and skip without it.

use std::path::PathBuf;

use serde_json::Value;
use tidelm_text::glm::{render, Channel, GlmIds, GlmParser};
use tidelm_text::Text;

fn cases() -> Value {
    serde_json::from_str(include_str!("glm_cases.json")).unwrap()
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("GLM_TOKENIZER_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap()).join("models/GLM-5.3-EXL3-3.0bpw")
    });
    dir.join("tokenizer.json").exists().then_some(dir)
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
fn prompts_match_glm_app() {
    let c = cases();
    let all = c["render"].as_array().unwrap();
    assert!(all.len() >= 50, "the fixture has {} render cases", all.len());
    for c in all {
        let name = c["name"].as_str().unwrap();
        let tools = Some(&c["tools"]).filter(|t| !t.is_null());
        let got = render(c["messages"].as_array().unwrap(), tools, c["effort"].as_str());
        match (got, c.get("error")) {
            (Ok(text), None) => assert_eq!(text, c["text"].as_str().unwrap(), "{name}"),
            (Err(_), Some(_)) => {}
            (got, want) => panic!("{name}: got {got:?}, jinja2 gave {want:?}"),
        }
    }
}

#[test]
fn prompt_ids_match_the_python_tokenizer() {
    let Some(dir) = model_dir() else {
        eprintln!("GLM tokenizer not present; skipping");
        return;
    };
    let text = Text::load(&dir).unwrap();
    let mut n = 0;
    for c in cases()["render"].as_array().unwrap().iter().filter(|c| c.get("ids").is_some()) {
        let got = text.encode(c["text"].as_str().unwrap()).unwrap();
        assert_eq!(got, ids(&c["ids"]), "{}", c["name"]);
        n += got.len();
    }
    assert!(n > 4_000, "{n} ids compared");
}

#[test]
fn the_parser_routes_and_extracts_as_glm_parser() {
    let Some(dir) = model_dir() else {
        eprintln!("GLM tokenizer not present; skipping");
        return;
    };
    let text = Text::load(&dir).unwrap();
    let c = cases();
    let gids = GlmIds::load(&dir, &text).unwrap();
    assert_eq!(gids, GlmIds { think: c["think"].as_u64().unwrap() as u32, think_end: c["think_end"].as_u64().unwrap() as u32,
                              call: c["call"].as_u64().unwrap() as u32, call_end: c["call_end"].as_u64().unwrap() as u32,
                              stop: ids(&c["stop"]) });
    let chan = |c: Channel| if c == Channel::Reasoning { "reasoning" } else { "content" };
    let events = |ev: Vec<(Channel, String)>| -> Value { ev.into_iter().map(|(c, t)| serde_json::json!([chan(c), t])).collect() };
    for c in c["parse"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if let Some(src) = c["source"].as_str() {
            assert_eq!(text.encode(src).unwrap(), ids(&c["ids"]), "{name}: ids");
        }
        let mut p = GlmParser::new(&gids);
        let mut ev = Vec::new();
        for t in ids(&c["ids"]) {
            ev.extend(p.feed(&text, t));
        }
        assert_eq!(events(ev), c["events"], "{name}: events");
        assert_eq!(events(p.finish(&text)), c["final"], "{name}: final");
        assert_eq!(p.reasoning, c["reasoning"].as_str().unwrap(), "{name}: reasoning");
        assert_eq!(p.content, c["content"].as_str().unwrap(), "{name}: content");
        assert_eq!(Value::from(p.call_texts(&text)), c["call_texts"], "{name}: call texts");
        let tools = Some(&c["tools"]).filter(|t| !t.is_null());
        let calls: Value = p.tool_calls(&text, tools).into_iter().map(|(n, a)| serde_json::json!([n, a])).collect();
        assert_eq!(calls, c["tool_calls"], "{name}: tool calls");
    }
}
