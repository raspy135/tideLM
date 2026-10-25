//! tidelm_text::chat against the checkpoint's encoding/encoding.py (encode_messages,
//! parse_message_from_completion_text) and server/app.py's tolerant tool-call parser, on the
//! cases tests/chat_tools_cases.json records (gen_chat_tools_cases.py runs the Python):
//! tools, namespaces, calls and results, list contents, tasks, images, errors.

use serde_json::Value;
use tidelm_text::chat::{encode_messages, parse_completion, parse_tool_calls_tolerant};

fn cases() -> Value {
    serde_json::from_str(include_str!("chat_tools_cases.json")).unwrap()
}

#[test]
fn prompts_match_the_checkpoint_encoder() {
    for c in cases()["encode"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let got = encode_messages(c["messages"].as_array().unwrap(), c["thinking"].as_bool().unwrap(),
                                  c["effort"].as_u64().unwrap() as u32);
        match (got, c.get("error")) {
            (Ok(e), None) => {
                assert_eq!(e.prompt, c["prompt"].as_str().unwrap(), "{name}");
                assert_eq!(Value::Array(e.images), c["images"], "{name}: images");
            }
            // a set's order is Python's own: compare up to it
            (Err(e), Some(want)) => {
                let want = want.as_str().unwrap();
                let cut = want.find(" Valid tasks are").unwrap_or(want.len());
                assert_eq!(&e[..cut.min(e.len())], &want[..cut], "{name}");
            }
            (got, want) => panic!("{name}: got {got:?}, the encoder gave {want:?}"),
        }
    }
}

#[test]
fn answers_parse_as_the_checkpoint_parser_and_the_tolerant_fallback_parse_them() {
    for c in cases()["parse"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let text = c["text"].as_str().unwrap();
        match (parse_completion(text, c["thinking"].as_bool().unwrap()), c.get("error")) {
            (Ok(p), None) => {
                let want = &c["parsed"];
                assert_eq!(p.content, want["content"].as_str().unwrap(), "{name}");
                assert_eq!(p.reasoning, want["reasoning"].as_str().unwrap(), "{name}");
                let calls: Vec<Value> = p
                    .tool_calls
                    .iter()
                    .map(|t| serde_json::json!([t.name, t.namespace, t.arguments]))
                    .collect();
                assert_eq!(Value::Array(calls), want["tool_calls"], "{name}");
            }
            (Err(e), Some(want)) => assert_eq!(e, want.as_str().unwrap(), "{name}"),
            (got, want) => panic!("{name}: got {got:?}, the parser gave {want:?}"),
        }
        let tolerant: Vec<Value> = parse_tool_calls_tolerant(text).into_iter().map(|(n, a)| serde_json::json!([n, a])).collect();
        assert_eq!(Value::Array(tolerant), c["tolerant"], "{name}: tolerant");
    }
}
