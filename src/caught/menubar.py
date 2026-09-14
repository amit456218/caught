"""macOS menu bar wrapper around the pipeline + server."""
from __future__ import annotations

import threading
import time
import webbrowser

from .config import HABITS, Config


def run_menubar(cfg: Config) -> int:
    import rumps

    from .capture import CameraSource
    from .focus import Sessions
    from .pipeline import Pipeline
    from .server import create_app, serve_in_thread
    from .store import Store

    cfg.ensure_dirs()
    store = Store(cfg.db_path, cfg.samples_path)
    pipe = Pipeline(cfg, CameraSource(cfg.camera_index, cfg.process_width), store, log=lambda *a: None)
    app_ = create_app(cfg, store, pipe)
    server, _ = serve_in_thread(app_, cfg.server_host, cfg.server_port)
    url = f"http://{cfg.server_host}:{cfg.server_port}"
    sessions = Sessions(store)

    class App(rumps.App):
        def __init__(self):
            super().__init__("👁", quit_button=None)
            self.status_item = rumps.MenuItem("starting…")
            self.pause_item = rumps.MenuItem("Pause watching", callback=self.toggle_pause)
            self.session_item = rumps.MenuItem("No focus session")
            self.menu = [
                self.status_item, self.session_item, None,
                rumps.MenuItem("Open dashboard", callback=lambda _: webbrowser.open(url)),
                rumps.MenuItem("Start 25 min session", callback=lambda _: sessions.start(25)),
                rumps.MenuItem("Start 50 min session", callback=lambda _: sessions.start(50)),
                rumps.MenuItem("End session", callback=lambda _: sessions.end()),
                None,
                self.pause_item,
                rumps.MenuItem("Calibrate posture (10s)", callback=lambda _: pipe.start_calibration(10)),
                rumps.MenuItem("Train model now", callback=lambda _: pipe.train_async("menubar")),
                None,
                rumps.MenuItem("Quit", callback=self.quit),
            ]
            pipe.start()
            threading.Thread(target=self._tick, daemon=True).start()

        def _tick(self):
            while True:
                st = pipe.status
                if not st.get("running"):
                    self.title = "👁"
                    self.status_item.title = "camera not running" + (f": {pipe.error}" if pipe.error else "")
                else:
                    active = [k for k, v in st.get("habits", {}).items() if v.get("active")]
                    if not st.get("present"):
                        self.title = "👁 ·"
                    elif active:
                        self.title = "👁 " + "".join(HABITS[k].icon for k in active if k != "absent")
                    else:
                        self.title = "👁"
                    self.status_item.title = (f"{'present' if st.get('present') else 'away'} · "
                                              f"{st.get('fps', 0)} fps · {st.get('mode')}")
                try:
                    live = sessions.live()
                except Exception:  # noqa: BLE001
                    live = None
                if live:
                    sm = live["summary"]
                    self.session_item.title = (f"Session {sm['elapsed_minutes']:.0f}/{live['planned_minutes']:.0f} min · "
                                               f"score {sm['score']} · {sm['phone_pickups']} pickups")
                else:
                    self.session_item.title = "No focus session"
                time.sleep(1.0)

        def toggle_pause(self, sender):
            pipe.pause(not pipe.paused)
            sender.title = "Resume watching" if pipe.paused else "Pause watching"

        def quit(self, _):
            pipe.stop()
            server.should_exit = True
            store.close()
            rumps.quit_application()

    App().run()
    return 0
