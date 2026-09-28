# nerf_web_remote.py
import asyncio
import json
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
from aiohttp import web

WEB_REMOTE_VERSION = "webremote-v3"
print(f"[web remote] module loaded: {__file__} | {WEB_REMOTE_VERSION}")

# ------------------------------------------------------------
# Simple phone-friendly page (drag to look, hold buttons to move)
# ------------------------------------------------------------

WEB_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no"/>
  <title>NeRF Training Remote</title>
  <style>
    /* Disable text selection + iOS long-press callout */
    html, body, #topbar, #hint, #padWrap, #pad, button, canvas, span {
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none; /* iOS: no copy/paste bubble */
    }
    /* Also prevent tap highlight + unwanted drag image */
    button, canvas {
    -webkit-tap-highlight-color: transparent;
    -webkit-user-drag: none;
    }

    body { margin:0; background:#111; color:#ddd; font-family: system-ui, sans-serif; overflow-x:hidden; }
    #topbar { padding:8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    button { padding:10px 14px; font-size:16px; border:0; border-radius:8px; background:#2a2a2a; color:#ddd; }
    button:active { background:#444; }
    .danger { background:#7a1f1f; }
    #wrap { display:flex; justify-content:center; align-items:center; padding:8px 0 90px; } /* bottom padding for pad */
    canvas { touch-action:none; image-rendering: pixelated; background:#000; }
    #hint { opacity:0.8; font-size:13px; padding:0 8px 8px; text-align:center; }
    /* bottom-fixed movement pad */
    #padWrap{
    position:fixed;
    left:0; right:0; bottom:0;
    padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    background:rgba(10,10,10,0.92);
    border-top:1px solid rgba(255,255,255,0.08);
    display:flex;
    justify-content:center;
    z-index:10;
    }
    #pad{
    display:grid;
    grid-template-columns: repeat(3, 70px);
    grid-template-rows: repeat(2, 56px);
    gap:10px;
    }
    /* slightly larger pad buttons */
    #pad button{
    font-size:18px;
    border-radius:12px;
    }
  </style>
</head>
<body>
  <div id="topbar">
    <button id="stop" class="danger">Stop training</button>
    <button id="save">Save frame</button>
    <span id="stat">disconnected</span>
    <span id="ver" style="opacity:0.7; font-size:12px;">{{WEB_REMOTE_VERSION}}</span>
  </div>

  <div id="wrap">
    <canvas id="cv" width="72" height="128" style="width:360px; height:640px;"></canvas>
  </div>

  <div id="hint">Drag = look. Hold buttons / WASDQE = move. Low-res frames scaled up.</div>

  <!-- Bottom movement pad -->
  <div id="padWrap">
    <div id="pad"></div>
  </div>

<script>
    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;

    let ws = null;
    let connected = false;

    let W = 72, H = 128; // overwritten by server
    let cam = { x:-2.0, y:0.0, z:0.5, yaw:0.0, pitch:0.0 };
    let keys = { w:false,a:false,s:false,d:false,q:false,e:false };
    let dragging = false;
    let lastX = 0, lastY = 0;

    function setStat(s){ document.getElementById("stat").textContent = s; }

    setStat("js started");

    window.onerror = (msg, src, line, col, err) => {
    setStat("JS error: " + msg + " @" + line + ":" + col);
    };
    window.onunhandledrejection = (e) => {
    setStat("Promise error: " + (e.reason ? e.reason.toString() : "unknown"));
    };

    function connectWS(){
    const proto = (location.protocol === "https:") ? "wss://" : "ws://";
    const wsUrl = proto + location.host + "/ws";
    setStat("connecting: " + wsUrl);

    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => { connected = true; setStat("connected"); };
    ws.onclose = () => { connected = false; setStat("disconnected"); setTimeout(connectWS, 800); };
    ws.onerror = () => { connected = false; setStat("ws error (see firewall?)"); };

    ws.onmessage = (ev) => {
        if (typeof ev.data === "string"){
            try{
                const msg = JSON.parse(ev.data);
                if (msg.type === "init" && msg.cam){
                    cam.x = msg.cam.x ?? cam.x;
                    cam.y = msg.cam.y ?? cam.y;
                    cam.z = msg.cam.z ?? cam.z;
                    cam.yaw = msg.cam.yaw ?? cam.yaw;
                    cam.pitch = msg.cam.pitch ?? cam.pitch;
                    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
                    setStat("init pose received");
                }
            }catch(e){
                // ignore
            }
            return;
        }
        const buf = new Uint8Array(ev.data);
        const dv = new DataView(buf.buffer);
        const w = dv.getUint32(0, true);
        const h = dv.getUint32(4, true);
        const step = dv.getUint32(8, true);
        const jpeg = buf.slice(12);

        if (w !== W || h !== H){
        W = w; H = h;
        cv.width = W; cv.height = H;
        }
        setStat(`step ${step} | ${W}x${H}`);

        const blob = new Blob([jpeg], {type:"image/jpeg"});
        const url = URL.createObjectURL(blob);
        const im = new Image();
        im.onload = () => {
        ctx.clearRect(0,0,W,H);
        ctx.drawImage(im, 0,0,W,H);
        URL.revokeObjectURL(url);
        };
        im.src = url;
    };
    }

    function sendState(moving=false, save=false, stop=false){
    if (!connected) return;
    ws.send(JSON.stringify({
        type: "state",
        cam: cam,
        moving: moving ? 1 : 0,
        save: save ? 1 : 0,
        stop: stop ? 1 : 0
    }));
    }

    document.getElementById("stop").onclick = () => sendState(false, false, true);
    document.getElementById("save").onclick = () => sendState(false, true, false);

    // --------------------
    // Drag-to-look (INVERTED so it feels like grabbing the world)
    // --------------------
    cv.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointerup",   () => { dragging = false; });
    cv.addEventListener("pointercancel",()=> { dragging = false; });

    cv.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;

    const sens = 0.006;

    // Inverted vs your old version:
    // old: cam.yaw -= dx*sens; cam.pitch -= dy*sens;
    cam.yaw   += dx * sens;
    cam.pitch += dy * sens;

    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
    sendState(true, false, false);
    });

    // --------------------
    // Bottom movement pad: Q W E / A S D
    // --------------------
    function mkHoldButton(label, key){
    const b = document.createElement("button");
    b.textContent = label;

    // Touch / mouse hold
    b.onpointerdown = (e) => { e.preventDefault(); keys[key]=true; };
    b.onpointerup = b.onpointercancel = b.onpointerleave = (e) => { e.preventDefault(); keys[key]=false; };

    return b;
    }

    const pad = document.getElementById("pad");

    // Order: top row Q W E, bottom row A S D
    pad.appendChild(mkHoldButton("Q","q"));
    pad.appendChild(mkHoldButton("W","w"));
    pad.appendChild(mkHoldButton("E","e"));
    pad.appendChild(mkHoldButton("A","a"));
    pad.appendChild(mkHoldButton("S","s"));
    pad.appendChild(mkHoldButton("D","d"));

    // --------------------
    // Keyboard controls (WASDQE)
    // --------------------
    function keyToControl(e){
    // Use e.code so it works regardless of keyboard layout
    switch (e.code){
        case "KeyW": return "w";
        case "KeyA": return "a";
        case "KeyS": return "s";
        case "KeyD": return "d";
        case "KeyQ": return "q";
        case "KeyE": return "e";
        default: return null;
    }
    }

    window.addEventListener("keydown", (e) => {
    const k = keyToControl(e);
    if (!k) return;
    // avoid page scrolling on some browsers
    e.preventDefault();
    keys[k] = true;
    });

    window.addEventListener("keyup", (e) => {
    const k = keyToControl(e);
    if (!k) return;
    e.preventDefault();
    keys[k] = false;
    });

    // If tab loses focus, release all keys to avoid “stuck key”
    window.addEventListener("blur", () => {
    for (const k in keys) keys[k] = false;
    });

    // --------------------
    // Integrate movement
    // --------------------
    let lastT = performance.now();
    setInterval(() => {
    const t = performance.now();
    const dt = Math.min(0.05, (t - lastT) / 1000.0);
    lastT = t;

    const spd = 1.0;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const fx = cy*cp, fy = sy*cp, fz = sp;

    // right = forward x up_world(0,0,1)
    const rx = fy;
    const ry = -fx;
    const rz = 0.0;
    const rn = Math.hypot(rx,ry,rz) || 1.0;
    const rnx = rx/rn, rny = ry/rn, rnz = rz/rn;

    // up = right x forward
    const ux = rny*fz - rnz*fy;
    const uy = rnz*fx - rnx*fz;
    const uz = rnx*fy - rny*fx;

    let vx=0, vy=0, vz=0;
    if (keys.w){ vx+=fx; vy+=fy; vz+=fz; }
    if (keys.s){ vx-=fx; vy-=fy; vz-=fz; }
    if (keys.d){ vx+=rnx; vy+=rny; vz+=rnz; }
    if (keys.a){ vx-=rnx; vy-=rny; vz-=rnz; }
    if (keys.e){ vx+=ux; vy+=uy; vz+=uz; }
    if (keys.q){ vx-=ux; vy-=uy; vz-=uz; }

    const vlen = Math.hypot(vx,vy,vz);
    const moving = vlen > 1e-6;
    if (moving){
        vx/=vlen; vy/=vlen; vz/=vlen;
        cam.x += spd*dt*vx;
        cam.y += spd*dt*vy;
        cam.z += spd*dt*vz;
        sendState(true,false,false);
    }
    }, 33);

    connectWS();
</script>
</body>

</html>
"""

def _yaw_pitch_from_forward(fwd_world: np.ndarray) -> tuple[float, float]:
    """
    Convert world-space forward direction to viewer's yaw/pitch convention:
      fx = cos(yaw)*cos(pitch)
      fy = sin(yaw)*cos(pitch)
      fz = sin(pitch)
    """
    f = np.asarray(fwd_world, dtype=np.float64).reshape(3)
    n = np.linalg.norm(f)
    if n < 1e-12:
        return 0.0, 0.0
    f = f / n
    yaw = float(np.arctan2(f[1], f[0]))
    pitch = float(np.arcsin(np.clip(f[2], -1.0, 1.0)))
    return yaw, pitch


def cam_state_from_c2w(c2w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    c2w: (4,4) camera-to-world.
    Camera axes are columns of R:
      col0 = e_u (right), col1 = e_v (down), col2 = e_w (forward).
    """
    T = np.asarray(c2w, dtype=np.float64)
    C = T[:3, 3].astype(np.float32)
    fwd = T[:3, 2]  # e_w in world
    yaw, pitch = _yaw_pitch_from_forward(fwd)
    return C, yaw, pitch


# ------------------------------------------------------------
# Server state + API
# ------------------------------------------------------------

@dataclass
class Controls:
    cam_pos: np.ndarray
    yaw: float
    pitch: float
    last_input_time: float
    moving_flag: int
    save_request: bool
    stop_request: bool

def make_preview_indices(W_cam: int, H_cam: int, down: int):
    """
    Returns (W_ds, H_ds, idx_flat, idy_flat) for center-of-block sampling.
    """
    W_cam = int(W_cam); H_cam = int(H_cam); down = int(down)
    W_ds, H_ds = W_cam // down, H_cam // down
    x_ds = np.tile(np.arange(W_ds, dtype=np.int32), H_ds)
    y_ds = np.repeat(np.arange(H_ds, dtype=np.int32), W_ds)
    idx_flat = np.clip(x_ds * down + (down // 2), 0, W_cam - 1).astype(np.int32)
    idy_flat = np.clip(y_ds * down + (down // 2), 0, H_cam - 1).astype(np.int32)
    return W_ds, H_ds, idx_flat, idy_flat


class WebRemoteServer:
    """
    aiohttp server in background thread.
    - pop_controls(): training thread reads controls
    - broadcast_jpeg(): training thread pushes frames
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = int(port)

        self._lock = threading.Lock()

        # Defaults (can be overwritten via set_initial_from_c2w / set_initial)
        self._cam_pos = np.array([-2.0, 0.0, 0.5], dtype=np.float32)
        self._yaw = 0.0
        self._pitch = 0.0

        self._last_input_time = 0.0
        self._moving_flag = 0
        self._save_request = False
        self._stop_request = False

        self._loop = None
        self._thread = None
        self._stop_evt = threading.Event()

        self._clients = set()  # websocket clients

    async def _index(self, request):
        html = WEB_PAGE.replace("{{WEB_REMOTE_VERSION}}", WEB_REMOTE_VERSION)
        return web.Response(
            text=html,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    async def _ws(self, request):
        peer = request.remote
        print(f"[web remote] ws connect from {peer}")
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        self._clients.add(ws)
        
        # Send initial camera state to client immediately
        with self._lock:
            init_cam = {
                "x": float(self._cam_pos[0]),
                "y": float(self._cam_pos[1]),
                "z": float(self._cam_pos[2]),
                "yaw": float(self._yaw),
                "pitch": float(self._pitch),
            }
        try:
            await ws.send_str(json.dumps({"type": "init", "cam": init_cam}))
        except Exception:
            pass

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception as e:
                    print("[web remote] bad json:", repr(e))
                    continue
                if data.get("type") != "state":
                    continue
                cam = data.get("cam", {})
                with self._lock:
                    self._cam_pos = np.array([cam.get("x",-2.0), cam.get("y",0.0), cam.get("z",0.5)], dtype=np.float32)
                    self._yaw = float(cam.get("yaw", 0.0))
                    self._pitch = float(cam.get("pitch", 0.0))
                    self._last_input_time = time.time()
                    self._moving_flag = int(data.get("moving", 0))
                    if int(data.get("save", 0)):
                        self._save_request = True
                    if int(data.get("stop", 0)):
                        self._stop_request = True
        except Exception as e:
            print("[web remote] ws handler error:", repr(e))
        finally:
            self._clients.discard(ws)
            print(f"[web remote] ws disconnect from {peer}")
        return ws


    async def _start_async(self):
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        # keep refs
        self._app = app
        self._runner = runner
        self._site = site

    async def _stop_async(self):
        try:
            for ws in list(self._clients):
                await ws.close()
        except Exception:
            pass
        self._clients.clear()
        if getattr(self, "_runner", None) is not None:
            await self._runner.cleanup()

    def set_initial(self, cam_pos: np.ndarray, yaw: float, pitch: float):
        """Set initial camera state (thread-safe)."""
        with self._lock:
            self._cam_pos = np.asarray(cam_pos, dtype=np.float32).reshape(3).copy()
            self._yaw = float(yaw)
            self._pitch = float(pitch)

    def set_initial_from_c2w(self, c2w: np.ndarray):
        """Set initial camera state from a c2w pose."""
        C, yaw, pitch = cam_state_from_c2w(c2w)
        self.set_initial(C, yaw, pitch)

    def start(self):
        ready = threading.Event()
        errors = []
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._start_async())
                ready.set()
                while not self._stop_evt.is_set():
                    self._loop.run_until_complete(asyncio.sleep(0.1))
            except Exception as error:
                errors.append(error)
                ready.set()
            finally:
                self._loop.run_until_complete(self._stop_async())
                self._loop.close()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=10):
            self._stop_evt.set()
            raise RuntimeError("Browser preview server startup timed out")
        if errors:
            raise RuntimeError(f"Cannot start browser preview on {self.host}:{self.port}") from errors[0]
        print(f"[web remote] open http://{self.host}:{self.port}/ (use your PC LAN IP on phone)")

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def pop_controls(self) -> Controls:
        """
        Returns latest control state and clears one-shot 'save'.
        stop_request remains latched until you exit or clear it manually.
        """
        with self._lock:
            save = bool(self._save_request)
            self._save_request = False
            return Controls(
                cam_pos=self._cam_pos.copy(),
                yaw=float(self._yaw),
                pitch=float(self._pitch),
                last_input_time=float(self._last_input_time),
                moving_flag=int(self._moving_flag),
                save_request=save,
                stop_request=bool(self._stop_request),
            )

    def broadcast_jpeg(self, jpeg_bytes: bytes, W: int, H: int, step: int):
        """
        Sends binary: [u32 W][u32 H][u32 step][jpeg...]
        """
        if self._loop is None or not self._clients:
            return

        header = struct.pack("<III", int(W), int(H), int(step))
        payload = header + jpeg_bytes

        async def _send_all():
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_bytes(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
        except Exception:
            pass
