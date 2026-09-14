"""Optional macOS notifications. Off by default: the product does not nag."""
from __future__ import annotations

import subprocess


def notify(title: str, body: str) -> None:
    script = f'display notification "{body}" with title "{title}"'
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
