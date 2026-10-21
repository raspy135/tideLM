#!/usr/bin/env python3
"""The dream driver: when the engine has been idle long enough, learn from the recent conversations.

    python3 dream/dream.py run              # the opt-in daemon: watch for idle time, dream at night
    python3 dream/dream.py now [--no-sleep] # dream right away (stage 1, then stage 2 unless --no-sleep)
    python3 dream/dream.py status | wake | skip | log [N]

Nothing here runs unless started; the engine itself only logs conversations and answers `@@`
commands when it was booted with DSV41_DREAM=1 (server/commands.py). A dream is two stages:

  1. prepare (engine up, idle >= idle_prepare_min): the served model reads the conversations since
     the last dream (at most `days` back) and writes tonight's fact list (dream/extract_facts.py),
     then a lesson set at the user's reasoning effort (build_lessons.py), the stay-close / drift
     text (eval_facts.py controls), and the training corpus (build_corpus.py). Fresh every night:
     only the weights carry over. Requests a user sends meanwhile queue behind at most one of these;
     the stage stops at the next step boundary if the engine is no longer idle and starts over later.
  2. sleep (inside the night window, idle >= idle_sleep_min, last dream >= min_gap_h ago): stop the
     engine, put dream/standin.py on its port (so clients see "dreaming", not "down"), run the
     prefix pass and training (dream/train.py, resumed from the last masters), and if the drift gate
     passes make the new generation current on both nodes (dream/overlay.py). Then the engine boots
     again with the current overlay. `@@dream wake` (or `dream.py wake`) stops training at the next
     step; its masters are saved and the next night resumes from them.

Every night leaves one record in DREAM_DIR/dreams.jsonl (what `@@dream log` shows), including the
nights it decided not to dream and why. Settings: DREAM_DIR/config.json over DEFAULTS below.
Standard library only (runs on the host; the GPU work runs in the tidelm image).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from server import commands as C  # noqa: E402

HOST_DIR = Path(os.environ.get("DREAM_HOST_DIR", "/home/ryan/models/dream"))   # this host's path
CONT_DIR = os.environ.get("DREAM_CONT_DIR", "/models/dream")                  # the same, in containers
# (a test area: DREAM_HOST_DIR=/home/ryan/models/dream-test DREAM_CONT_DIR=/models/dream-test -- it must
# live under the models dir, the only host path the engine container sees)

DEFAULTS = {
    "url": "http://127.0.0.1:8000",
    "profile": ".env.ds41-mxfp4",
    "boot": ["scripts/tide-up.sh", "{profile}"],     # how the engine is started again after a dream
    "image": "tidelm:local",
    "models_dir": "/home/ryan/models",
    "window": ["01:00", "07:00"],                     # stage 2 may only start inside this
    "latest_start": "03:00",                          # ...and no later than this
    "idle_prepare_min": 30,
    "idle_sleep_min": 60,
    "min_gap_h": 20,
    "days": 3,
    "max_facts": 50,
    "effort": "medium",                               # the user serves thinking on at medium
    "lessons_per_fact": 12,
    # forward "tiers": trained through the FP8 copy a fresh lesson is served from (the engine's FP8
    # tier, tools/fp4_moe_cuda.cu dream_fp8_*) plus 0.3 x the MXFP4 copy it consolidates into. 2026-10-21's
    # test of "mix" (master + 0.3 MXFP4, lr 5e-5) left MXFP4 exactly stock after 4 epochs.
    "train": {"first": 36, "experts": 8, "lr": 1e-4, "epochs": 6, "batch": 8, "eval_every": 2,
              "forward": "tiers", "beta": 0.3, "kl_weight": 1.0, "max_drift": 0.05,
              "fp8_budget": 64},
    "keep_nights": 3,                                 # training dirs kept (each holds ~13 GB of masters)
}


# ------------------------------------------------------------------ plumbing
def config() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    p = HOST_DIR / "config.json"
    if p.exists():
        over = json.loads(p.read_text())
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(HOST_DIR / "dream.log", "a") as f:
        f.write(line + "\n")


def write_state(**s):
    s = {**s, "updated": time.time()}
    tmp = HOST_DIR / "state.json.tmp"
    tmp.write_text(json.dumps(s))
    os.replace(tmp, HOST_DIR / "state.json")
    return s


def clear_state():
    write_state(state="awake")


def record(**r):
    with open(HOST_DIR / "dreams.jsonl", "a") as f:
        f.write(json.dumps({"date": C.night_of().isoformat(), **r}, ensure_ascii=False) + "\n")


def last_record(outcomes=None) -> dict | None:
    for r in reversed(C.dream_records(1000)):
        if outcomes is None or r.get("outcome") in outcomes:
            return r
    return None


def health(url, timeout=3) -> dict | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def last_activity() -> float:
    """Newest conversation-log write: the engine logs every finished user completion (and never
    the dream's own requests), so this is the last time a person used it."""
    d = HOST_DIR / "conversations"
    return max((p.stat().st_mtime for p in d.glob("*.jsonl")), default=0.0) if d.is_dir() else 0.0


def idle_min() -> float:
    return (time.time() - last_activity()) / 60


def run(cmd, log_path=None, env=None, cwd=ROOT) -> int:
    log("$ " + " ".join(map(str, cmd)))
    with open(log_path or os.devnull, "a") as out:
        return subprocess.call(list(map(str, cmd)), stdout=out, stderr=subprocess.STDOUT, cwd=cwd,
                               env={**os.environ, **(env or {})})


def hm_today(s: str) -> dt.datetime:
    h, m = map(int, s.split(":"))
    return dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)


def in_window(cfg) -> bool:
    now = dt.datetime.now()
    lo, hi, latest = hm_today(cfg["window"][0]), hm_today(cfg["window"][1]), hm_today(cfg["latest_start"])
    return lo <= now <= min(hi, latest) if lo <= hi else (now >= lo or now <= min(hi, latest))


def wake_requested() -> bool:
    return (HOST_DIR / "wake.request").exists()


def clear_wake():
    try:
        (HOST_DIR / "wake.request").unlink()
    except FileNotFoundError:
        pass


def tonight() -> Path:
    d = HOST_DIR / "nights" / C.night_of().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cont(p: Path) -> str:
    """The container path of a file under HOST_DIR."""
    return CONT_DIR + "/" + str(Path(p).resolve().relative_to(HOST_DIR.resolve()))


# ------------------------------------------------------------------ stage 1
def prepare(cfg, force=False, fresh=False) -> dict | None:
    """Write tonight's corpus. Returns its description, or None (nothing to learn / interrupted).
    force: skip the idle checks (`now`). fresh: redo it even if tonight's corpus exists."""
    night = tonight()
    done = night / "prepared.json"
    if done.exists() and not fresh:
        return json.loads(done.read_text())
    url = cfg["url"]
    prev = last_record(("served", "gate failed", "woken"))
    since = prev["until_ts"] if prev and prev.get("until_ts") else time.time() - cfg["days"] * 86400
    until = time.time()
    plog = night / "prepare.log"
    write_state(state="awake", stage="preparing", started=time.time())

    def still_idle():
        if force:
            return True
        ok = idle_min() >= cfg["idle_prepare_min"]
        if not ok:
            log("stage 1 paused: someone is using the engine; it starts over when it is idle again")
        return ok

    facts = night / "facts.jsonl"
    rc = run([sys.executable, HERE / "extract_facts.py", "--out", facts, "--since", since,
              "--days", cfg["days"], "--max-facts", cfg["max_facts"], "--effort", cfg["effort"],
              "--conversations", HOST_DIR / "conversations", "--url", url], plog)
    if rc != 0:
        raise RuntimeError(f"extract_facts failed (see {plog})")
    n = sum(1 for line in open(facts) if line.strip())
    if n == 0:
        record(outcome="skipped", reason="nothing new to learn", since=_day(since), until=_day(until),
               until_ts=until, ended=time.time())
        done.write_text(json.dumps({"n_facts": 0}))
        return None
    if not still_idle():
        return None
    lessons = night / "lessons.jsonl"
    for extra in ([], ["--heldout"]):
        if run([sys.executable, HERE / "build_lessons.py", "--facts-file", facts, "--out", lessons,
                "--n", cfg["lessons_per_fact"], "--effort", cfg["effort"], "--url", url, *extra], plog):
            raise RuntimeError(f"build_lessons failed (see {plog})")
        if not still_idle():
            return None
    general = night / "general.jsonl"
    if run([sys.executable, HERE / "eval_facts.py", "run", "--set", "controls", "--effort", cfg["effort"],
            "--out", general, "--url", url], plog):
        raise RuntimeError(f"eval_facts controls failed (see {plog})")
    corpus = night / "corpus.jsonl"
    if run([sys.executable, HERE / "build_corpus.py", "--facts", facts, "--lessons", lessons, "--general", general,
            "--effort", cfg["effort"], "--out", corpus, "--url", url], plog):
        raise RuntimeError(f"build_corpus failed (see {plog})")
    desc = {"n_facts": n, "facts": str(facts), "corpus": str(corpus), "since": since, "until": until,
            "n_seqs": sum(1 for _ in open(corpus))}
    done.write_text(json.dumps(desc))
    log(f"stage 1 done: {n} facts, {desc['n_seqs']} sequences")
    return desc


def _day(ts):
    return time.strftime("%m-%d", time.localtime(ts)) if ts else "?"


# ------------------------------------------------------------------ stage 2
def profile_model_dir(cfg) -> str:
    """The serving checkpoint, as a container path. Read from the profile, never left to the image:
    tidelm:local sets MODEL_DIR=/models/DeepSeek-V4.1-Flash (the pre-1017 checkpoint), and every
    training run before 2026-10-21 trained on it while serving 1017 (docs/dreaming.md)."""
    prof = Path(cfg["profile"]) if Path(cfg["profile"]).is_absolute() else ROOT / cfg["profile"]
    for line in prof.read_text().splitlines():
        m = re.match(r"\s*MODEL_DIR=(\S+)", line)
        if m:
            host = Path(m.group(1).strip("\"'"))
            return "/models/" + str(host.relative_to(cfg["models_dir"]))
    raise RuntimeError(f"{prof} sets no MODEL_DIR")


def docker_train(cfg, args, name, log_path) -> subprocess.Popen:
    # --user: as root the trainer left train/ and serve/ owned by root, and the host-side
    # `overlay.py use` could not make serve/current (2026-10-21's first full dream).
    cmd = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--ipc", "host",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-e", f"MODEL_DIR={profile_model_dir(cfg)}",
           "-v", f"{ROOT}:/w:ro", "-v", f"{cfg['models_dir']}:/models:ro", "-v", f"{HOST_DIR}:{CONT_DIR}",
           "-w", "/w", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1", "-e", f"DREAM_DIR={CONT_DIR}",
           "--entrypoint", "python", cfg["image"], "dream/train.py", *args]
    log("$ " + " ".join(map(str, cmd)))
    out = open(log_path, "a")
    proc = subprocess.Popen(list(map(str, cmd)), stdout=out, stderr=subprocess.STDOUT)
    return proc


def watch(proc, name, log_path, state, of=None) -> str:
    """Wait for a training container; returns "done", "failed" or "woken"."""
    t0 = time.time()
    while proc.poll() is None:
        if wake_requested():
            log("wake requested: stopping training (masters are saved first)")
            subprocess.call(["docker", "stop", "-t", "900", name], stdout=subprocess.DEVNULL)
            proc.wait()
            return "woken"
        ep = _last_epoch(log_path)
        eta = None
        if of and ep:
            eta = time.time() + (time.time() - t0) / ep * (of - ep)
        write_state(**{**state, "epoch": ep if of else None, "of": of, "eta": eta or state.get("eta")})
        time.sleep(10)
    return "done" if proc.returncode == 0 else "failed"


def _last_epoch(log_path) -> int | None:
    try:
        eps = re.findall(r" epoch (\d+):", Path(log_path).read_text())
    except OSError:
        return None
    return int(eps[-1]) if eps else None


def sleep_stage(cfg, prep) -> None:
    t = cfg["train"]
    night = tonight()
    gen = "dream-" + C.night_of().strftime("%Y%m%d")
    runp = HOST_DIR / "train" / gen
    tlog = night / "train.log"
    prev = last_record(("served", "woken", "gate failed"))
    resume = prev.get("masters") if prev else None
    parent = (os.path.basename(os.path.realpath(HOST_DIR / "serve" / "current"))
              if (HOST_DIR / "serve" / "current").exists() else None)
    started = time.time()
    # rough: ~50-80 s a layer for the prefix pass, ~10 sequences a minute a epoch for training
    eta = started + 70 * t["first"] + 60 * prep["n_seqs"] * t["epochs"] / 10
    base = dict(state="dreaming", started=started, eta=eta, facts=prep["n_facts"],
                since=_day(prep["since"]), until=_day(prep["until"]))
    write_state(**base, stage="stopping the engine")
    clear_wake()
    run([ROOT / "scripts" / "tide-down.sh"], night / "engine.log")
    standin = subprocess.Popen([sys.executable, HERE / "standin.py", "--port",
                                cfg["url"].rsplit(":", 1)[-1].split("/")[0]],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    outcome, reason, gate, masters = "failed", "", {}, resume
    try:
        write_state(**base, stage="prefix pass")
        proc = docker_train(cfg, ["prefix", "--corpus", cont(Path(prep["corpus"])), "--run", cont(runp),
                                  "--first", t["first"]], "dream-train", tlog)
        how = watch(proc, "dream-train", tlog, {**base, "stage": "prefix pass"})
        if how != "done":
            outcome, reason = ("woken", "woken during the prefix pass") if how == "woken" else \
                ("failed", f"prefix pass failed (see {tlog})")
            return
        args = ["train", "--corpus", cont(Path(prep["corpus"])), "--run", cont(runp), "--first", t["first"],
                "--experts", t["experts"], "--lr", t["lr"], "--epochs", t["epochs"], "--batch", t["batch"],
                "--eval-every", t["eval_every"], "--forward", t["forward"], "--beta", t["beta"],
                "--kl-weight", t["kl_weight"], "--max-drift", t["max_drift"], "--fp8-budget", t["fp8_budget"],
                "--gen", gen, "--note", f"dream {C.night_of()}: {prep['n_facts']} facts"]
        if resume:
            args += ["--resume", resume] + (["--parent", parent] if parent else [])
        proc = docker_train(cfg, args, "dream-train", tlog)
        how = watch(proc, "dream-train", tlog, {**base, "stage": "training"}, of=t["epochs"])
        if how == "woken":
            part = runp / f"masters-{gen}.partial.pt"
            masters = cont(part) if part.exists() else resume
            outcome, reason = "woken", "woken by @@dream wake; progress kept for the next dream"
            return
        if how != "done":
            outcome, reason = "failed", f"training failed (see {tlog})"
            return
        masters = cont(runp / f"masters-{gen}.pt")
        man = json.loads((HOST_DIR / "serve" / gen / "manifest.json").read_text())
        gate = man.get("drift_gate", {})
        if not gate.get("pass"):
            outcome, reason = "gate failed", f"drift {gate.get('drift')} over {gate.get('max_drift')}"
            return
        write_state(**base, stage="switching the overlay")
        oenv = {"DREAM_DIR": str(HOST_DIR)}
        if run([sys.executable, HERE / "overlay.py", "use", gen], night / "overlay.log", env=oenv) or \
                run([sys.executable, HERE / "overlay.py", "push"], night / "overlay.log", env=oenv):
            outcome, reason = "failed", "could not switch or copy the overlay"
            return
        outcome = "served"
    finally:
        standin.terminate()
        standin.wait()
        write_state(state="booting", stage="booting the engine")
        env = {"DSV41_DREAM": "1", "DSV41_DREAM_DIR": CONT_DIR}
        if (HOST_DIR / "serve" / "current").exists():
            env["DSV41_DREAM_OVERLAY"] = CONT_DIR + "/serve/current"
        boot = [c.replace("{profile}", cfg["profile"]) for c in cfg["boot"]]
        rc = run([ROOT / boot[0], *boot[1:]], night / "engine.log", env=env)
        clear_wake()
        clear_state()
        facts = [json.loads(line) for line in open(prep["facts"])] if prep.get("facts") else []
        record(outcome=outcome, reason=reason, started=started, ended=time.time(), generation=gen,
               parent=parent, masters=masters, since=_day(prep["since"]), until=_day(prep["until"]),
               until_ts=prep["until"],
               n_facts=prep["n_facts"], facts=[{k: f.get(k) for k in ("kind", "seen", "fact")} for f in facts],
               drift=gate.get("drift"), max_drift=gate.get("max_drift"), gate_pass=gate.get("pass"),
               engine_boot=rc)
        log(f"dream {outcome}{': ' + reason if reason else ''}; engine boot exit {rc}")
        housekeeping(cfg)


def housekeeping(cfg):
    """Keep the last keep_nights training dirs; older ones go (they are root-owned: the trainer
    container wrote them, so a container removes them). The masters a resume needs are kept."""
    need = {r.get("masters") for r in C.dream_records(1000)[-2:]}
    runs = sorted((HOST_DIR / "train").glob("dream-*"))
    for d in runs[:-cfg["keep_nights"]]:
        if any(m and cont(d) in m for m in need):
            continue
        subprocess.call(["docker", "run", "--rm", "-v", f"{HOST_DIR}:{CONT_DIR}", "--entrypoint", "rm",
                         cfg["image"], "-rf", cont(d)])
        log(f"removed old training dir {d}")


# ------------------------------------------------------------------ commands
def cmd_run(a, cfg):
    log("dream daemon started")
    while True:
        try:
            tick(cfg)
        except Exception as e:  # noqa: BLE001
            log(f"error: {e!r}")
            clear_state()
        time.sleep(60)
        cfg = config()


def tick(cfg):
    if C.tonight_skipped():
        return
    last = last_record()
    if last and last.get("ended") and time.time() - last["ended"] < cfg["min_gap_h"] * 3600:
        return
    h = health(cfg["url"])
    if not h or h.get("busy"):
        return
    idle = idle_min()
    if idle >= cfg["idle_prepare_min"]:
        prep = prepare(cfg)
        if prep and prep.get("n_facts") and in_window(cfg) and idle_min() >= cfg["idle_sleep_min"]:
            sleep_stage(cfg, prep)


def cmd_now(a, cfg):
    prep = prepare(cfg, force=True, fresh=a.fresh)
    if prep and prep.get("n_facts") and not a.no_sleep:
        sleep_stage(cfg, prep)


def cmd_status(a, cfg):
    print(C.format_state(C.dream_state()))
    for r in C.dream_records(3):
        print("  " + C.summary_line(r))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    n = sub.add_parser("now")
    n.add_argument("--no-sleep", action="store_true", help="stage 1 only")
    n.add_argument("--fresh", action="store_true", help="redo stage 1 even if tonight's corpus exists")
    sub.add_parser("status")
    sub.add_parser("wake")
    sub.add_parser("skip")
    lg = sub.add_parser("log")
    lg.add_argument("n", nargs="?", type=int, default=1)
    a = p.parse_args()
    os.environ.setdefault(C.DIR_ENV, str(HOST_DIR))
    HOST_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config()
    if a.cmd in ("wake", "skip"):
        print(C.run("dream", [a.cmd], "dreaming" if C.dream_state().get("state") == "dreaming" else "awake")
              .lstrip(C.MARKER))
        return
    if a.cmd == "log":
        print(C.run("dream", ["log", str(a.n)], "awake").lstrip(C.MARKER))
        return
    {"run": cmd_run, "now": cmd_now, "status": cmd_status}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
