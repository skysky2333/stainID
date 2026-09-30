"""Download the public model weights stainID uses (Cellpose-SAM, Segment Anything ViT-B, Phikon) into the project."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from stainid.project import Project

URLS = {
    "cellpose": "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2",
    "sam": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
PUBLIC = ("cellpose", "sam", "huggingface_home")


def _fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        total, done, shown = int(response.headers.get("Content-Length", 0)), 0, -1
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            percent = int(100 * done / total) if total else 0
            if percent != shown and percent % 5 == 0:
                print(f"[{percent}/100] {target.name}: {done / 1e6:.0f} MB", flush=True)
                shown = percent
    partial.rename(target)


def download_models(project: Project, which: list[str] | None = None) -> None:
    for key in which or PUBLIC:
        target = project.model(key)
        if key == "huggingface_home":
            from huggingface_hub import snapshot_download

            os.environ["HF_HUB_OFFLINE"] = "0"
            print(f"Downloading Phikon (owkin/phikon) into {project.relative(target)}", flush=True)
            snapshot_download("owkin/phikon", cache_dir=target / "hub")
        elif target.exists():
            print(f"{key}: already present at {project.relative(target)}", flush=True)
        else:
            print(f"Downloading {key} weights to {project.relative(target)}", flush=True)
            _fetch(URLS[key], target)
    print("Done", flush=True)
