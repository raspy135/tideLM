//! tidelm_text::chat against corpus/sources/dsv41_encoding.py::encode_messages on recorded
//! cases (tests/chat_cases.json, written by the Python encoder).

use tidelm_text::chat::{encode, Message, Role};

#[test]
fn chat_prompts_match_the_python_encoder() {
    let cases: serde_json::Value = serde_json::from_str(include_str!("chat_cases.json")).unwrap();
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let msgs: Vec<Message> = c["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    _ => Role::Assistant,
                },
                content: m["content"].as_str().unwrap().to_string(),
                reasoning: m.get("reasoning_content").and_then(|r| r.as_str()).map(String::from),
            })
            .collect();
        let effort = c["effort"].as_u64().map(|e| e as u32);
        let got = encode(&msgs, c["thinking"].as_bool().unwrap(), effort).unwrap();
        assert_eq!(got, c["prompt"].as_str().unwrap(), "case {i}");
    }
}
