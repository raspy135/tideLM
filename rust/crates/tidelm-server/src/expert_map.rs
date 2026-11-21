//! GET /expert-map (the page) and GET /v1/expert-map (its data): which routed experts are resident
//! now, for every engine this server runs.
//!
//! The engine thread answers Cmd::Map between its steps with a copy of host data
//! (tidelm_engine::expert_map); this module turns it into JSON on the connection's thread. An engine
//! in a long step (a prefill chunk) answers when it ends: past WAIT the last answer goes out again,
//! marked stale, so the page keeps drawing. Nothing is asked of the engine unless a page polls.

use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tidelm_engine::expert_map::ExpertMap;

use crate::api::Failure;
use crate::http::{send_json, ApiError};
use crate::{Cmd, Ctx, Family};

const WAIT: Duration = Duration::from_millis(1500);

pub const PAGE: &str = include_str!("expert_map.html");

/// What a page asks for, from the query: `since` the last route it holds and `events_since` the last event (only
/// newer ones are sent: a page polls five times a second), `demand=0` (the usage rows left out, a fast poll's).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ask {
    pub since: u64,
    pub events_since: u64,
    pub demand: bool,
}

impl Ask {
    pub fn parse(query: &str) -> Ask {
        let mut a = Ask { demand: true, ..Ask::default() };
        for (k, v) in query.split('&').filter_map(|kv| kv.split_once('=')) {
            match k {
                "since" => a.since = v.parse().unwrap_or(0),
                "events_since" => a.events_since = v.parse().unwrap_or(0),
                "demand" => a.demand = v != "0",
                _ => {}
            }
        }
        a
    }
}

/// The snapshot as the page reads it. `states`: a string a layer, a digit an expert (0 not loaded,
/// 1 resident, 2 in the hot-read ring, 3 being read); `demand`: the same shape in base 36, an
/// expert's share of its layer's routing history (0 none, z the layer's most used; null when not asked for,
/// `has_demand` says whether the engine keeps one). `events`: the engine's recent residency events after
/// `events_since`, oldest first, each [seq, age ms, row, expert, kind, share in thousandths, victim row,
/// victim expert] (-1, -1 without a victim; `event_kinds` names the kinds) - what brought each expert in: a
/// missed pick (its share of the token's routing weight), an inline rescue, an adaptation, a preload.
/// `routes`: the decoded tokens' routing after `since`, oldest first, each {seq, age_ms, lane, pos, token (its
/// text, to a keyed request only: it is the conversation's; else null), k, experts [row * k + pick], share
/// (thousandths, the same order), missed (a digit a pick: 1 when it was not resident as the pass ran)}.
pub fn json(m: &ExpertMap, model: &str, engine: &str, ask: &Ask, token: Option<&dyn Fn(u32) -> String>) -> serde_json::Value {
    let c = m.counts();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let row = |l: u16| m.layers.binary_search(&(l as usize)).map(|r| r as i64).unwrap_or(-1);
    let events: Vec<serde_json::Value> = m.events.iter().filter(|e| e.seq > ask.events_since && row(e.layer) >= 0).map(|e| {
        let (vr, vx) = e.victim.map(|(l, x)| (row(l), x as i64)).unwrap_or((-1, -1));
        serde_json::json!([e.seq, ((now - e.at) * 1000.0).max(0.0).round() as i64, row(e.layer), e.expert, e.kind,
                           (e.share * 1000.0).round() as i64, vr, vx])
    }).collect();
    let routes: Vec<serde_json::Value> = m.routes.iter().filter(|r| r.seq > ask.since && r.picks.len() == m.layers.len() * r.topk as usize).map(|r| {
        serde_json::json!({
            "seq": r.seq, "age_ms": ((now - r.at) * 1000.0).max(0.0).round() as i64, "lane": r.lane, "pos": r.pos,
            "token": token.map(|f| f(r.token)), "k": r.topk,
            "experts": r.picks.iter().map(|p| p.0).collect::<Vec<_>>(),
            "share": r.picks.iter().map(|p| (p.1 * 1000.0).round() as i64).collect::<Vec<_>>(),
            "missed": r.picks.iter().map(|p| if p.2 { '1' } else { '0' }).collect::<String>(),
        })
    }).collect();
    let resident_per_layer: Vec<usize> =
        m.state.chunks(m.experts.max(1)).map(|r| r.iter().filter(|&&s| s == tidelm_engine::expert_map::RESIDENT).count()).collect();
    let counters: serde_json::Map<String, serde_json::Value> = m.counters.iter().map(|&(k, v)| (k.to_string(), v.into())).collect();
    serde_json::json!({
        "version": 3,
        "model": model,
        "engine": engine,
        "sparks": m.world,
        "busy": m.busy,
        "stale": false,
        "sampled_at": now,
        "layers": m.layers,
        "experts_per_layer": m.experts,
        "total_experts": m.state.len(),
        "state_names": ["not_loaded", "resident", "hot", "loading"],
        "counts": {"not_loaded": c[0], "resident": c[1], "hot": c[2], "loading": c[3]},
        "resident_per_layer": resident_per_layer,
        "arena_slots": m.slots,
        "spare_slots": m.spare,
        "counters": counters,
        "states": m.state_rows(),
        "has_demand": m.demand.is_some(),
        "demand": if ask.demand { serde_json::json!(m.demand_rows()) } else { serde_json::Value::Null },
        "events": events,
        "event_seq": m.events.last().map(|e| e.seq).unwrap_or(0).max(ask.events_since),
        "event_kinds": tidelm_engine::expert_map::EV_NAMES,
        "routes": routes,
        "route_seq": m.routes.last().map(|r| r.seq).unwrap_or(0).max(ask.since),
        "tokens_shown": token.is_some(),
    })
}

/// `keyed`: the request carried one of the server's keys, or the server wants none (the routed tokens' text goes
/// only then).
pub fn serve(stream: &mut TcpStream, ctx: &Ctx, query: &str, keyed: bool) -> Result<(), Failure> {
    let ask = Ask::parse(query);
    let (tx, rx) = mpsc::channel();
    let gone = || Failure::Api(ApiError { status: 503, message: "the engine is not running".into(), param: None });
    ctx.engine.send(Cmd::Map(tx)).map_err(|_| gone())?;
    let engine = match &ctx.family {
        Family::Ds => "deepseek",
        Family::Glm(_) => "glm",
    };
    let text = |id: u32| ctx.text.decode(&[id]).unwrap_or_default();
    let v = match rx.recv_timeout(WAIT) {
        Ok(Some(m)) => {
            let v = json(&m, &ctx.model, engine, &ask, keyed.then_some(&text as &dyn Fn(u32) -> String));
            // kept for a poll that finds the engine mid-step: without the routes' text, whoever asks next
            let mut kept = v.clone();
            if let Some(rs) = kept["routes"].as_array_mut() {
                for r in rs {
                    r["token"] = serde_json::Value::Null;
                }
            }
            kept["tokens_shown"] = false.into();
            *ctx.map_last.lock().unwrap() = Some(kept);
            v
        }
        Ok(None) => return Err(Failure::Api(ApiError { status: 404, message: "this engine holds every expert: there is no residency to map".into(), param: None })),
        Err(mpsc::RecvTimeoutError::Disconnected) => return Err(gone()),
        // the engine is inside a step: the last answer again, said to be old, with nothing new in it
        Err(mpsc::RecvTimeoutError::Timeout) => match ctx.map_last.lock().unwrap().clone() {
            Some(mut v) => {
                v["stale"] = true.into();
                v["busy"] = true.into();
                v["routes"] = serde_json::json!([]);
                v["events"] = serde_json::json!([]);
                v["route_seq"] = ask.since.into();
                v["event_seq"] = ask.events_since.into();
                v
            }
            None => return Err(Failure::Api(ApiError { status: 503, message: "the engine is busy: no snapshot yet".into(), param: None })),
        },
    };
    send_json(stream, 200, &v).map_err(|_| Failure::Io)
}

#[cfg(test)]
mod tests {
    use super::Ask;

    #[test]
    fn the_query_asks() {
        assert_eq!(Ask::parse(""), Ask { since: 0, events_since: 0, demand: true });
        assert_eq!(Ask::parse("since=41&events_since=7&demand=0"), Ask { since: 41, events_since: 7, demand: false });
        assert_eq!(Ask::parse("since=x&demand=1&other"), Ask { since: 0, events_since: 0, demand: true });
    }
}
