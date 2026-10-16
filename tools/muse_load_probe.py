"""Load the engine and exit (memory instrumentation with MUSE_MEMLOG=1)."""
import os
import sys
sys.path.insert(0, "/app")
from engine.muse.tp import MuseTP, mem_available, rss_anon  # noqa: E402
eng = MuseTP(os.environ.get("MUSE_MODEL", "/models"))
print(f"[mem r{eng.rank}] loaded: MemAvailable {mem_available() / 1e9:.1f} GB, {rss_anon() / 1e9:.1f} GB anon", flush=True)
