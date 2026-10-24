//! The DeepSeek-V4.1 chat prompt (corpus/sources/dsv41_encoding.py::encode_messages), for
//! text messages: system (leading and mid-conversation), user and assistant turns, chat and
//! thinking modes with the numeric reasoning effort, and earlier turns' reasoning dropped.
//! Tool definitions, tool calls, tasks and images are not rendered here; a request carrying
//! them is refused by the caller rather than encoded differently.

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const SYSTEM: &str = "<｜System｜>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// an assistant turn's reasoning (rendered only for turns after the last user message)
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message { role, content: content.into(), reasoning: None }
    }
}

/// The index of the last user turn; a mid-conversation system message counts as one.
fn last_user(messages: &[Message]) -> Option<usize> {
    (0..messages.len()).rev().find(|&i| messages[i].role == Role::User || (messages[i].role == Role::System && i > 0))
}

/// encode_messages(messages, thinking_mode, reasoning_effort) with drop_thinking and the
/// default BOS: `thinking` selects "thinking" over "chat"; `effort` is the 1..=100 budget
/// (None: "high", 75), rendered only in thinking mode.
pub fn encode(messages: &[Message], thinking: bool, effort: Option<u32>) -> Result<String, String> {
    if let Some(e) = effort {
        if !(1..=100).contains(&e) {
            return Err(format!("reasoning effort {e} outside 1..=100"));
        }
    }
    // drop_thinking: before the last user turn only user/system turns and the assistant turns
    // (their reasoning removed) remain - every role here, so only the reasoning goes
    let last = last_user(messages);
    let msgs: Vec<Message> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut m = m.clone();
            if thinking && m.role == Role::Assistant && last.is_some_and(|l| i < l) {
                m.reasoning = None;
            }
            m
        })
        .collect();
    let mut out = String::from(BOS);
    for i in 0..msgs.len() {
        out.push_str(&render(i, &msgs, last, thinking, effort.unwrap_or(75)));
    }
    Ok(out)
}

fn render(i: usize, msgs: &[Message], last: Option<usize>, thinking: bool, effort: u32) -> String {
    let m = &msgs[i];
    let effort_prompt = if i == 0 && thinking {
        format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
    } else {
        String::new()
    };
    let mut p = String::new();
    if i == 0 && (!effort_prompt.is_empty() || m.role == Role::System) {
        p.push_str(SYSTEM);
    }
    p.push_str(&effort_prompt);
    match m.role {
        Role::System => {
            if i > 0 {
                p.push_str(SYSTEM);
            }
            p.push_str(&m.content);
        }
        Role::User => {
            p.push_str(USER);
            p.push_str(&m.content);
        }
        Role::Assistant => {
            // drop_thinking: only turns after the last user turn carry their reasoning
            if thinking && last.is_none_or(|l| i > l) {
                p.push_str(m.reasoning.as_deref().unwrap_or(""));
                p.push_str(THINK_END);
            }
            p.push_str(&m.content);
            p.push_str(EOS);
        }
    }
    // the generation header follows a user (or mid-conversation system) turn that is not
    // followed by an assistant turn
    if i + 1 < msgs.len() && msgs[i + 1].role != Role::Assistant {
        return p;
    }
    if m.role == Role::User || (m.role == Role::System && i > 0) {
        p.push_str(ASSISTANT);
        p.push_str(if thinking && last.is_some_and(|l| i >= l) { THINK_START } else { THINK_END });
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_mode_single_user_turn_is_the_bench_prompt() {
        let p = encode(&[Message::new(Role::User, "[req 1] hi")], false, None).unwrap();
        assert_eq!(p, "<｜begin▁of▁sentence｜><｜User｜>[req 1] hi<｜Assistant｜></think>");
    }

    #[test]
    fn thinking_mode_renders_the_effort_under_a_system_token() {
        let p = encode(&[Message::new(Role::User, "q")], true, Some(60)).unwrap();
        assert_eq!(p, "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 60 (range 1-100, the higher the value, the more \
                       thorough the reasoning)\n\n<｜User｜>q<｜Assistant｜><think>");
    }
}
