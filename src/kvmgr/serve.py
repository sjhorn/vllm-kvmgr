"""kvmgr.serve -- publish the counters from the process that holds them.

WHY ITS OWN PORT, and the two reasons are different.

1. *** IT MUST BE THE ENGINE CORE PROCESS. ***  The counters live where the KV
   managers live, which is the engine core, behind the IPC that carries
   EngineCoreOutputs.  A stat logger runs on AsyncLLM's output-handler loop in
   the API server process and cannot see them -- so the plan's "export via the
   stat_logger entry point" would report zeros forever.  This server is started
   from the manager's own module, so it is in the right process by construction
   rather than by hope.

2. NOT ON PORT 8888.  The proxy's Caddy catch-all forwards every path to the
   vLLM failover pool, so any route added to the serving app is reachable by any
   mTLS client and any LAN client on :443.  prefill_collector records the same
   reasoning for `/prefill`; sparkstats -- already tunnelled, already admin-only
   -- is what proxies this.

OFF BY DEFAULT, AND BY BINDING NOTHING.  `kvmgr_stats_port` is 0 unless set,
which means no socket exists at all.  That is this repo's own convention for an
opt-in capability -- `METER_TAP_PORT` means it by having no default port -- and
it is a stronger statement than a route that answers 404 because a boolean said
so.

IT MUST NOT BE ABLE TO HARM A REQUEST.  Daemon thread, bounded work, every
handler total.  Nothing here touches a request, a block or a token.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm.logger import init_logger

from . import flags, stats

logger = init_logger(__name__)

_server: ThreadingHTTPServer | None = None
_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the default stderr spam
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/kv"):
                body = json.dumps(stats.snapshot(), separators=(",", ":")).encode()
                self._send(200, body, "application/json")
            elif path == "/metrics":
                self._send(200, stats.prometheus().encode(), "text/plain; version=0.0.4")
            elif path == "/flags":
                # What the engine ACTUALLY has, which is the question a flag
                # flip raises and which reading the file cannot answer -- env
                # wins over it, silently.
                body = json.dumps(
                    {
                        "effective": {n: flags.raw(n) for n in flags.DEFAULTS},
                        "defaults": flags.DEFAULTS,
                        "env_pinned": flags.audit_env(),
                        "path": _flags_path(),
                    },
                    default=str, separators=(",", ":"),
                ).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b'{"error":"not found"}', "application/json")
        except Exception:  # never let a reader take the engine down
            try:
                self._send(500, b'{"error":"internal"}', "application/json")
            except Exception:
                pass


def _flags_path() -> str:
    try:
        from vllm.v1.core.single_type_kv_cache_manager import _FLAGS_PATH

        return _FLAGS_PATH
    except Exception:
        return "?"


def maybe_start() -> None:
    """Start the server if a port is configured. Idempotent and total.

    Called from the manager's constructor rather than from register(), so it
    starts in whichever process actually builds managers -- which is the whole
    point and is not knowable at plugin-load time.
    """
    global _server
    if _server is not None:
        return
    try:
        port = int(flags.raw("kvmgr_stats_port") or 0)
    except Exception:
        port = 0
    if port <= 0:
        return
    with _lock:
        if _server is not None:
            return
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
            srv.daemon_threads = True
            threading.Thread(
                target=srv.serve_forever, name="kvmgr-stats", daemon=True
            ).start()
            _server = srv
            logger.info(
                "kvmgr: stats on 127.0.0.1:%d (/kv json, /metrics prometheus, "
                "/flags effective) -- engine-core process, loopback only",
                port,
            )
        except Exception:
            # A port collision must not cost the engine. It costs observability,
            # and says so once.
            logger.exception(
                "kvmgr: could not bind the stats port %d; counters are still "
                "recorded but nothing can read them", port,
            )
