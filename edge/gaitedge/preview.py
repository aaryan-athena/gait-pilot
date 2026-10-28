"""A live view of what the device sees, served as a web page.

The service runs as a system account with no access to a desktop, so it
cannot open a window on an attached screen. Instead it publishes the camera
image, with what it is doing drawn on, at ``http://<host>:<port>/``; a
launcher in the desktop session (``deploy/preview_kiosk.sh``) opens that page
when a monitor is connected.

It costs the camera loop very little: frames are downscaled and JPEG-encoded
at a few frames a second, not at the camera's rate, and only while someone is
watching. Recording always comes first.

By default it listens on 127.0.0.1 only, so the camera cannot be watched from
the network. ``preview.host: 0.0.0.0`` opens it to the local network, which is
handy for aiming a camera but should not be left on in a care setting.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

import numpy as np

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>gaitedge live view</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#0e1a1f;color:#e8f0f2;
    font-family:Inter,-apple-system,"Segoe UI",sans-serif}
  .wrap{display:flex;flex-direction:column;height:100%}
  .bar{display:flex;gap:1.6rem;align-items:center;padding:.6rem 1.1rem;
    background:#13262d;font-size:.95rem;flex-wrap:wrap}
  .brand{font-weight:800;letter-spacing:.02em}
  .pill{padding:.15rem .6rem;border-radius:999px;font-weight:700;font-size:.85rem}
  .watch{background:#1f3b45;color:#9fd3cb}
  .rec{background:#b42318;color:#fff;animation:blink 1s steps(2,start) infinite}
  @keyframes blink{to{opacity:.55}}
  .muted{color:#8aa3aa}
  .stage{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
  img{max-width:100%;max-height:100%;object-fit:contain}
  .msg{color:#8aa3aa;font-size:1.1rem}
</style></head><body><div class="wrap">
  <div class="bar">
    <span class="brand">gaitedge</span>
    <span id="device" class="muted"></span>
    <span id="state" class="pill watch">connecting…</span>
    <span id="motion" class="muted"></span>
    <span id="queue" class="muted"></span>
    <span id="last" class="muted"></span>
  </div>
  <div class="stage"><img id="feed" alt="live camera">
    <div id="msg" class="msg" style="display:none">Waiting for the camera service…</div></div>
</div>
<script>
const feed=document.getElementById('feed'), msg=document.getElementById('msg');
function connect(){feed.src='/stream?'+Date.now();}
feed.onload=()=>{msg.style.display='none';feed.style.display='';};
feed.onerror=()=>{feed.style.display='none';msg.style.display='';setTimeout(connect,2000);};
connect();
async function poll(){
  try{
    const s=await (await fetch('/status',{cache:'no-store'})).json();
    document.getElementById('device').textContent=s.device||'';
    const st=document.getElementById('state');
    if(s.recording){st.className='pill rec';st.textContent='● REC '+s.recording_s.toFixed(0)+' s';}
    else{st.className='pill watch';st.textContent=s.state||'watching';}
    document.getElementById('motion').textContent=
      s.motion==null?'':'motion '+(100*s.motion).toFixed(1)+'%';
    const q=s.queue||{};
    document.getElementById('queue').textContent=
      'waiting '+((q.pending||0)+(q.processing||0))+' · done '+(q.done||0);
    document.getElementById('last').textContent=s.last_result?('last: '+s.last_result):'';
  }catch(e){document.getElementById('state').textContent='service not running';}
  setTimeout(poll,1000);
}
poll();
</script></body></html>"""


class Preview:
    """Latest annotated frame plus status, served over HTTP."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8080, fps: float = 8.0,
                 width: int = 960, status: Optional[Callable[[], dict]] = None):
        self.fps = float(fps)
        self.width = int(width)
        self._status_fn = status or (lambda: {})
        self._jpeg: Optional[bytes] = None
        self._cond = threading.Condition()
        self._last_encode = 0.0
        self._viewers = 0
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, int(port)), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="preview", daemon=True)
        self._thread.start()

    # -- fed by the camera loop ----------------------------------------------
    def offer(self, image: np.ndarray, *, recording_s: Optional[float] = None,
              motion: Optional[float] = None) -> None:
        """Called for every camera frame; encodes only a few per second."""
        now = time.monotonic()
        # Nobody watching: no work at all. Otherwise at most `fps` encodes.
        if (self._viewers == 0 and self._jpeg is not None) or \
                now - self._last_encode < 1.0 / self.fps:
            return
        self._last_encode = now
        jpeg = _annotate_and_encode(image, self.width, recording_s, motion)
        with self._cond:
            self._jpeg = jpeg
            self._cond.notify_all()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # -- HTTP ------------------------------------------------------------------
    def _handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # the service log is for walks, not page loads
                pass

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/":
                    return self._send(200, "text/html; charset=utf-8", PAGE.encode())
                if path == "/status":
                    body = json.dumps(preview._status_fn(), default=str).encode()
                    return self._send(200, "application/json", body)
                if path == "/snapshot.jpg":
                    if preview._jpeg is None:
                        return self._send(503, "text/plain", b"no frame yet")
                    return self._send(200, "image/jpeg", preview._jpeg)
                if path == "/stream":
                    return self._stream()
                self._send(404, "text/plain", b"not found")

            def _send(self, status, ctype, body):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with preview._lock:
                    preview._viewers += 1
                try:
                    last = None
                    while True:
                        with preview._cond:
                            preview._cond.wait_for(lambda: preview._jpeg is not last,
                                                   timeout=5.0)
                            jpeg = preview._jpeg
                        if jpeg is None or jpeg is last:
                            continue
                        last = jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                         + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with preview._lock:
                        preview._viewers -= 1

        return Handler


def _annotate_and_encode(image: np.ndarray, width: int, recording_s: Optional[float],
                         motion: Optional[float]) -> bytes:
    import cv2

    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(round(h * width / w))),
                           interpolation=cv2.INTER_AREA)
    else:
        image = image.copy()
    if recording_s is not None:
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1),
                      (24, 35, 180), 6)
        label = f"REC {recording_s:4.0f} s"
        cv2.circle(image, (26, 30), 9, (24, 35, 220), -1)
        cv2.putText(image, label, (44, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    if motion is not None:
        # A small meter of what the motion trigger sees, bottom left.
        bar = int(min(1.0, motion / 0.10) * 160)
        y = image.shape[0] - 18
        cv2.rectangle(image, (14, y - 8), (174, y + 2), (60, 60, 60), -1)
        cv2.rectangle(image, (14, y - 8), (14 + bar, y + 2), (99, 200, 160), -1)
        cv2.putText(image, "motion", (182, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else b""
