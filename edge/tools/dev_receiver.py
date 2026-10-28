"""A minimal server implementing the gaitedge upload API, for development.

    python tools/dev_receiver.py --port 8700 --store ./received --token devtoken

Not the product server: no database, no web pages, no analysis of raw clips.
It exists so the device's upload path can be exercised end to end before the
real platform is built, and as an executable statement of the contract in
``gaitedge/uploader.py`` that the real server must honour -- in particular
that a repeated ``Idempotency-Key`` is acknowledged, not stored twice.

Standard library only, so it runs anywhere Python does.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def make_handler(store: Path, token: str | None, fail_first: int = 0):
    lock = threading.Lock()
    failures = {"left": int(fail_first)}

    class Handler(BaseHTTPRequestHandler):
        server_version = "gaitedge-dev-receiver/1"

        def log_message(self, fmt, *args):  # quieter than the default
            print(f"[receiver] {self.command} {self.path} -> {fmt % args}")

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            if not token:
                return True
            return self.headers.get("Authorization") == f"Bearer {token}"

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/api/v1/walks":
                walks = []
                for d in sorted((store / "walks").glob("*")):
                    manifest = json.loads((d / "manifest.json").read_text())
                    walks.append({"walk_id": d.name, "files": sorted(p.name for p in d.iterdir()),
                                  "processed_on": manifest.get("processed_on")})
                return self._json(200, {"walks": walks})
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._authorised():
                return self._json(401, {"error": "bad token"})
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)

            match = re.fullmatch(r"/api/v1/devices/([^/]+)/heartbeat", self.path)
            if match:
                device = _SAFE.sub("_", match.group(1))
                (store / "devices").mkdir(parents=True, exist_ok=True)
                (store / "devices" / f"{device}.json").write_bytes(body)
                return self._json(200, {"ok": True})

            if self.path != "/api/v1/walks":
                return self._json(404, {"error": "not found"})

            with lock:
                if failures["left"] > 0:
                    failures["left"] -= 1
                    return self._json(503, {"error": "simulated outage"})

            key = self.headers.get("Idempotency-Key")
            if not key:
                return self._json(400, {"error": "Idempotency-Key header required"})
            walk_id = _SAFE.sub("_", key)
            folder = store / "walks" / walk_id
            with lock:
                if folder.exists():
                    return self._json(409, {"walk_id": walk_id, "duplicate": True})
                parts = _parse_multipart(self.headers.get("Content-Type", ""), body)
                if "manifest" not in parts:
                    return self._json(400, {"error": "manifest part required"})
                tmp = folder.with_name(walk_id + ".partial")
                tmp.mkdir(parents=True, exist_ok=True)
                for name, (filename, data) in parts.items():
                    target = name + (".json" if name in ("manifest", "result") else
                                     Path(filename or "").suffix)
                    (tmp / target).write_bytes(data)
                tmp.rename(folder)
            return self._json(201, {"walk_id": walk_id, "received_at": time.time()})

    return Handler


def _parse_multipart(content_type: str, body: bytes) -> dict:
    message = BytesParser(policy=default_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    parts = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            parts[name] = (part.get_filename(), part.get_payload(decode=True))
    return parts


def serve(port: int, store: Path, token: str | None, fail_first: int = 0):
    store.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, token, fail_first))
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--store", type=Path, default=Path("received"))
    parser.add_argument("--token", default=None)
    parser.add_argument("--fail-first", type=int, default=0,
                        help="answer the first N uploads with 503, to exercise retries")
    args = parser.parse_args()
    server = serve(args.port, args.store, args.token, args.fail_first)
    print(f"listening on http://127.0.0.1:{args.port}, storing in {args.store}")
    server.serve_forever()


if __name__ == "__main__":
    main()
