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

*** AND IT IS NO LONGER PURELY READ-ONLY -- ONE POST EXISTS, and the exception
is argued rather than assumed. ***  `POST /kv/verify` is the lever for after
somebody has touched files on ONE node: a partial `kvsweep.sh`, an operator
`rm`, a restored backup.  It has to live here because it needs the disk index,
which is in THIS process -- the same `disk: null` trap that makes `GET /kv` on
:8888 answer about the wrong process -- and because :8888 is behind Caddy's
catch-all, so a mutating route there is reachable by any holder of a client
certificate.  It was attached to :8888 first and was structurally dead: 409 on
every call, because the router is built in the API server and the manager is
not there.

Two properties keep the exception narrow, and both are load-bearing:

  * it MUTATES NO MANAGER STATE.  It rebinds one module global and returns 202;
    the work happens on the scheduler's own thread next step.  `_wanted` and
    `_verified` have exactly one writer, which is the race the stall watchdog
    already paid for.
  * *** A CHANGE HERE IS NOT HOT, AND THE REASON IS THE RELOAD FAMILY AGAIN. ***
    `ThreadingHTTPServer` stamps `_Handler` into `RequestHandlerClass` at bind
    time, and `_server` is in `control._PRESERVE` -- correctly, since a reload
    must not leak a second listener -- so a reload builds a new `_Handler` that
    nothing serves and the running server keeps the old one.  Same shape as
    `all_managers` and the serializer shim: anything stamped once at boot is
    unrecoverable by a reload.  Editing this handler costs a RESTART, which is
    a reason to keep it small rather than a problem to work around.
  * it is the ONLY non-GET path.  Anything else answers 404, and this handler
    is not a place to grow a control plane -- that is what :8888's router is
    for, for everything that does not need the index.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm.logger import init_logger

from . import flags, stats

# *** THE LOGGER NAME IS LOAD-BEARING, and `__name__` is the wrong one. ***
#
# vLLM's logging config gives a handler to the `vllm` logger ONLY, with
# propagate=false, and configures no root handler. So a logger named
# `kvmgr.flags` has NO handler and an EFFECTIVE LEVEL OF 30 (WARNING) --
# every logger.info() it makes is silently dropped. Measured in the running
# container 2026-10-07:
#
#   init_logger("kvmgr.register")       handlers [] effective 30  -> dropped
#   init_logger("vllm.kvmgr.register")  handlers [] effective 20  -> printed
#
# That cost us the whole "did it take?" story on the first deploy: the
# registration line, the cap-on-change line, the flag-change line and the env
# audit were all invisible, which is indistinguishable from the plugin never
# having loaded. Naming the logger under `vllm.` makes it inherit that handler.
logger = init_logger("vllm." + __name__)

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


    # The request body cap. A control lever on a loopback listener in the
    # engine core reads a bounded number of bytes or it is a way to make the
    # engine core allocate; 4 KiB is three orders of magnitude more than the
    # largest sensible body here.
    MAX_BODY = 4096

    def do_POST(self):  # noqa: N802
        """`/kv/verify` only. Total, bounded, and it reports what it accepted.

        202 rather than 200, because the work has not happened: the pump
        applies it on the scheduler's thread, and *** the step runs only when
        there are requests ***, so on an idle engine this sits pending until
        traffic arrives. The response says so, and names where to read the
        outcome, because a lever whose effect cannot be confirmed is one
        nobody will trust.
        """
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path != "/kv/verify":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except Exception:
                n = 0
            if n > self.MAX_BODY:
                self._send(413, json.dumps({
                    "error": "body too large",
                    "max_bytes": self.MAX_BODY}).encode(), "application/json")
                return
            body = {}
            if n > 0:
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except Exception:
                    self._send(400, b'{"error":"body is not JSON"}',
                               "application/json")
                    return
            if not isinstance(body, dict):
                self._send(400, b'{"error":"body must be a JSON object"}',
                           "application/json")
                return

            from . import diskspec

            if diskspec._LIVE_MANAGER is None:
                # The listener starts from the manager's constructor, so this
                # is close to unreachable -- and it is reported rather than
                # assumed away, because an empty success here would read as a
                # verification that never happened.
                self._send(409, json.dumps({
                    "ok": False,
                    "error": ("no disk manager in this process yet; the "
                              "listener outlived the index, which should not "
                              "happen -- do not read this as verified"),
                }).encode(), "application/json")
                return
            requeue_all = bool(body.get("all"))
            clear_fault = bool(body.get("clear_fault"))
            if not requeue_all and not clear_fault:
                self._send(400, json.dumps({
                    "ok": False,
                    "error": "nothing asked for",
                    "accepts": {
                        "all": ("re-queue every indexed key for a cross-rank "
                                "re-check; bounded by VERIFY_WANTED_MAX, and "
                                "it does NOT unverify, so no lookup slows "
                                "while it runs"),
                        "clear_fault": ("drop a recorded verification fault "
                                        "so presence can be claimed again -- "
                                        "deliberately manual"),
                    }}).encode(), "application/json")
                return
            req = diskspec.request_verify(requeue_all=requeue_all,
                                          clear_fault=clear_fault)
            logger.warning(
                "kvmgr: POST /kv/verify accepted seq=%s all=%s clear_fault=%s "
                "-- queued for the scheduler thread, NOT applied yet",
                req.get("seq"), requeue_all, clear_fault)
            self._send(202, json.dumps({
                "ok": True,
                "accepted": req,
                "applied": False,
                "note": ("the pump applies this on the scheduler's thread. "
                         "THE STEP RUNS ONLY WHEN THERE ARE REQUESTS, so an "
                         "idle engine applies it when traffic next arrives. "
                         "Confirm with GET /kv -> verify.last_request, "
                         "matching this seq; verify.pending_request is "
                         "non-null until then."),
                "cost": ("`all` makes the pump issue blocking collective_rpc "
                         "calls from the scheduler step, one per "
                         "disk_verify_chunk keys, ~1054ms per 4000 measured. "
                         "The runtime rate is 0 by default for that reason."),
            }, default=str).encode(), "application/json")
        except Exception:  # never let a writer take the engine down either
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
                "kvmgr: stats on 127.0.0.1:%d (GET /kv json, /metrics "
                "prometheus, /flags effective; POST /kv/verify re-check) -- "
                "engine-core process, loopback only",
                port,
            )
        except Exception:
            # A port collision must not cost the engine. It costs observability,
            # and says so once.
            logger.exception(
                "kvmgr: could not bind the stats port %d; counters are still "
                "recorded but nothing can read them", port,
            )
