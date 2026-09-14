"""Fetch the backtesting media: Mixkit clips (Mixkit license) and Wikimedia Commons photos.

    uv run python bench/fetch.py

Downloads into bench/videos and bench/commons, then links everything into bench/all next to
the labeled manifest. Re-runnable; skips files that already exist.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (caught bench fetch)"}

MIXKIT = {  # id: local name
    14101: "hands_phone_desk", 33743: "office_yawn_portrait", 51396: "sleepy_girl_phone",
    14821: "tired_student_library", 6255: "tired_student_college", 13316: "boy_asleep_desk",
    36104: "girl_book_falls_asleep", 9190: "tired_boy_studying", 44063: "man_working_computer",
    23720: "man_coffee_computer", 23724: "woman_coffee_laptop", 29989: "man_laptop_study",
    47572: "woman_yawn_pink", 51386: "woman_laptop_bed_asleep", 42628: "woman_on_call_working",
    21560: "staff_eating_desk", 4799: "boy_rubs_eyes", 41174: "man_texting_happy",
    41165: "man_answering_whatsapp", 4837: "man_street_typing_phone",
}


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    print(f"  {dest.name}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def main() -> int:
    videos, commons, all_dir = HERE / "videos", HERE / "commons", HERE / "all"
    videos.mkdir(exist_ok=True)
    commons.mkdir(exist_ok=True)
    all_dir.mkdir(exist_ok=True)
    print("Mixkit clips:")
    for vid, name in MIXKIT.items():
        get(f"https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4", videos / f"{name}.mp4")
    sources = json.loads((commons / "sources.json").read_text())
    print("Wikimedia Commons photos:")
    for name, meta in sources.items():
        title = meta["title"].replace("File:", "")
        url = ("https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/"
               + urllib.request.quote(title) + "&width=1280")
        try:
            get(url, commons / name)
        except Exception as e:  # noqa: BLE001
            print(f"  failed {name}: {e}")
    for src in list(videos.glob("*.mp4")) + list(commons.glob("*.jpg")):
        link = all_dir / src.name
        if not link.exists():
            os.symlink(os.path.relpath(src, all_dir), link)
    print("done. run: uv run caught bench bench/all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
