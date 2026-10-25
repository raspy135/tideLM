//! Image inputs on their way in (image_processor.load_image_bytes and prepare_vl_inputs): an
//! image record's bytes - base64 `data`, an Anthropic `source`, a `data:` URL or an http(s)
//! URL - the tower's input (tidelm_image, Pillow's pixels), and the prompt's placeholders
//! expanded into spans. The Python reader also opens any other string as a local file; a
//! server on the network does not read its own disk for a client, so that is refused here.

use serde_json::Value;
use tidelm_engine::vision::{span_keys, ImageIn};
use tidelm_image::{prepare, Plan};

/// What the server needs of the checkpoint's vision settings.
pub struct VisionCtx {
    pub plan: Plan,
    pub image_token_id: u32,
}

/// A fetched image's limit: urlopen has none, a server should
const MAX_FETCH: u64 = 64 << 20;

/// base64.b64decode: characters outside the alphabet are skipped (validate=False); missing
/// padding is tolerated
fn b64(s: &str) -> Result<Vec<u8>, String> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let body = s.split('=').next().unwrap_or("");
    let digits: Vec<u32> = body.bytes().filter_map(val).collect();
    if digits.len() % 4 == 1 {
        return Err("base64: incorrect padding".into());
    }
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut v = 0u32;
        for (i, &d) in chunk.iter().enumerate() {
            v |= d << (18 - 6 * i);
        }
        let bytes = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
        out.extend_from_slice(&bytes[..chunk.len() - 1]);
    }
    Ok(out)
}

fn fetch(url: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(30)).build();
    let resp = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let mut out = Vec::new();
    resp.into_reader().take(MAX_FETCH + 1).read_to_end(&mut out).map_err(|e| format!("{url}: {e}"))?;
    if out.len() as u64 > MAX_FETCH {
        return Err(format!("{url}: larger than {} MB", MAX_FETCH >> 20));
    }
    Ok(out)
}

/// load_image_bytes: `data`, then `source` (its data, or its url), then `url`.
pub fn load_bytes(rec: &Value) -> Result<Vec<u8>, String> {
    if let Some(Value::String(d)) = rec.get("data") {
        return b64(d);
    }
    if let Some(src) = rec.get("source").and_then(|s| s.as_object()) {
        match src.get("data") {
            Some(Value::String(d)) => return b64(d),
            Some(v) if !v.is_null() => return Err("an image source's data must be base64 text".into()),
            _ => {}
        }
        if let Some(Value::String(u)) = src.get("url").filter(|u| u.as_str().is_some_and(|u| !u.is_empty())) {
            return load_bytes(&serde_json::json!({ "url": u }));
        }
    }
    match rec.get("url") {
        Some(Value::String(u)) if !u.is_empty() => {
            if let Some(rest) = u.strip_prefix("data:") {
                let (header, payload) = rest.split_once(',').unwrap_or((rest, ""));
                if !header.contains(";base64") {
                    return Err(format!("Unsupported data URL encoding: data:{header}"));
                }
                return b64(payload);
            }
            if u.starts_with("http://") || u.starts_with("https://") {
                return fetch(u);
            }
            Err("an image URL must be a data: or http(s) URL".into())
        }
        _ => Err(format!("Cannot load image from record: {:?}", rec.as_object().map(|o| o.keys().collect::<Vec<_>>()))),
    }
}

/// A prompt with its images in place: every placeholder a span of `image_token_id`, the
/// images' patches, and the prefix-cache key.
pub struct Expanded {
    pub ids: Vec<u32>,
    pub images: Vec<ImageIn>,
    pub key: Option<Vec<u32>>,
}

/// prepare_vl_inputs: each placeholder in `ids` takes the next image of `records` (in prompt
/// order, as the encoder collected them).
pub fn expand(ids: &[u32], records: &[Value], v: &VisionCtx) -> Result<Expanded, String> {
    let found = ids.iter().filter(|&&t| t == v.image_token_id).count();
    if found != records.len() {
        return Err(format!("Found {found} image tokens but got {} images", records.len()));
    }
    if records.is_empty() {
        return Ok(Expanded { ids: ids.to_vec(), images: Vec::new(), key: None });
    }
    let mut prepared = Vec::with_capacity(records.len());
    for (i, rec) in records.iter().enumerate() {
        let bytes = load_bytes(rec).map_err(|e| format!("image {i}: {e}"))?;
        prepared.push(prepare(&bytes, &v.plan).map_err(|e| format!("image {i}: {e}"))?);
    }
    let (mut out, mut key, mut images) = (Vec::new(), Vec::new(), Vec::new());
    let mut next = prepared.into_iter();
    for &t in ids {
        if t != v.image_token_id {
            out.push(t);
            key.push(t);
            continue;
        }
        let p = next.next().unwrap();
        let n = p.tokens();
        key.extend(span_keys(p.digest(), n));
        images.push(ImageIn { start: out.len(), vit_h: p.vit_h, vit_w: p.vit_w, patches: p.patches });
        out.extend(std::iter::repeat_n(v.image_token_id, n));
    }
    Ok(Expanded { ids: out, images, key: Some(key) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_as_python_reads_it() {
        assert_eq!(b64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(b64("aGVs\nbG8").unwrap(), b"hello");
        assert_eq!(b64("aGVsbA==").unwrap(), b"hell");
        assert!(b64("aGVsb").is_err());
        let rec = serde_json::json!({"url": "data:image/png;base64,aGk="});
        assert_eq!(load_bytes(&rec).unwrap(), b"hi");
        assert!(load_bytes(&serde_json::json!({"url": "/etc/passwd"})).is_err());
        assert!(load_bytes(&serde_json::json!({"url": "data:text/plain,hi"})).unwrap_err().starts_with("Unsupported data URL"));
    }
}
