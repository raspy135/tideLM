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

/// The snapshot as the page reads it. `states`: a string a layer, a digit an expert (0 not loaded,
/// 1 resident, 2 in the hot-read ring, 3 being read); `demand`: the same shape in base 36, an
/// expert's share of its layer's routing history (0 none, z the layer's most used).
pub fn json(m: &ExpertMap, model: &str, engine: &str) -> serde_json::Value {
    let c = m.counts();
    let resident_per_layer: Vec<usize> =
        m.state.chunks(m.experts.max(1)).map(|r| r.iter().filter(|&&s| s == tidelm_engine::expert_map::RESIDENT).count()).collect();
    let counters: serde_json::Map<String, serde_json::Value> = m.counters.iter().map(|&(k, v)| (k.to_string(), v.into())).collect();
    serde_json::json!({
        "version": 2,
        "model": model,
        "engine": engine,
        "sparks": m.world,
        "busy": m.busy,
        "stale": false,
        "sampled_at": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
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
        "demand": m.demand_rows(),
    })
}

pub fn serve(stream: &mut TcpStream, ctx: &Ctx) -> Result<(), Failure> {
    let (tx, rx) = mpsc::channel();
    let gone = || Failure::Api(ApiError { status: 503, message: "the engine is not running".into(), param: None });
    ctx.engine.send(Cmd::Map(tx)).map_err(|_| gone())?;
    let engine = match &ctx.family {
        Family::Ds => "deepseek",
        Family::Glm(_) => "glm",
    };
    let v = match rx.recv_timeout(WAIT) {
        Ok(Some(m)) => {
            let v = json(&m, &ctx.model, engine);
            *ctx.map_last.lock().unwrap() = Some(v.clone());
            v
        }
        Ok(None) => return Err(Failure::Api(ApiError { status: 404, message: "this engine holds every expert: there is no residency to map".into(), param: None })),
        Err(mpsc::RecvTimeoutError::Disconnected) => return Err(gone()),
        // the engine is inside a step: the last answer again, said to be old
        Err(mpsc::RecvTimeoutError::Timeout) => match ctx.map_last.lock().unwrap().clone() {
            Some(mut v) => {
                v["stale"] = true.into();
                v["busy"] = true.into();
                v
            }
            None => return Err(Failure::Api(ApiError { status: 503, message: "the engine is busy: no snapshot yet".into(), param: None })),
        },
    };
    send_json(stream, 200, &v).map_err(|_| Failure::Io)
}
