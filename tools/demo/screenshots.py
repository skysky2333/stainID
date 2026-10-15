"""Take the README screenshots of a running stainID web app (normally on the synthetic demo project).

    python tools/demo/screenshots.py http://127.0.0.1:18555 docs/images [--only viewer annotate]

Needs Playwright (`pip install playwright && python -m playwright install chromium`, best in a separate virtual
environment; stainID itself does not need it). Start the app first, e.g. on a project made by make_demo_project.py
and with a throwaway HOME, so that neither the list of recent projects nor the admin password of your own stainID
settings changes:

    HOME=/tmp/stainid-demo-home stainid --project /tmp/demo_project serve --no-browser --host 127.0.0.1 --port 18555

(add `--projects-dir` with a folder holding the demo project and a project made by make_setup_demo.py for a Projects
page with more than one study). The script signs in as admin, registers an annotator name for the annotation page,
opens each page of the project the server was started with (at /p/<project>/...) in the light theme and writes
projects.png, workflow.png, home.png, models.png, viewer.jpg, labels.png, annotate.jpg, results.png and settings.png.
The core shown in the viewer is the AD core with the highest tau+ neuron density (the spot with the most detections
near its centre), and the annotation page opens the field the annotator has started but not finished.
"""
from __future__ import annotations

import argparse
import secrets
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

SHOTS = {  # name: (path in the project, or None when chosen from the data, viewport width, height)
    "projects.png": ("", 1400, 640),  # the Projects page (the server's, not a project's)
    "workflow.png": ("/workflow", 1500, 1011),
    "home.png": ("/", 1500, 911),
    "models.png": ("/models", 1500, 911),
    "viewer.jpg": (None, 1500, 1000),
    "labels.png": ("/labels", 1400, 900),
    "annotate.jpg": (None, 1500, 1000),
    "results.png": ("/results", 1500, 911),
    "settings.png": ("/settings", 1500, 911),
}
REPLACE_TEXT = """([root, label]) => {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeValue.includes(root)) node.nodeValue = node.nodeValue.split(root).join(label)
  }
}"""


def settle(page: Page, extra: float = 1.0) -> None:
    """Wait until the page's requests have finished (the app polls every few seconds, so idle periods do occur)."""
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    page.wait_for_function("() => [...document.images].every((i) => i.complete)", timeout=60000)
    time.sleep(extra)


def viewer_target(page: Page, api: str, stain: str) -> str:
    rows = page.request.get(f"{api}/tables/results_core.csv/rows").json()["rows"]
    column = {"AT8": "tau_neuron_density_mm2", "6E10": "v2_plaque_density_mm2", "NeuN": "neun_profile_density_mm2"}[stain]
    best = max((r for r in rows if r.get(column) is not None), key=lambda r: (r.get("disease_group") == "AD", r[column]))
    core_id = best["core_id"]
    image = next(i for i in page.request.get(f"{api}/cores/{core_id}").json()["images"] if i["stain"] == stain)
    w, h = image["native_width_px"], image["native_height_px"]
    spots = [(int(w * fx), int(h * fy)) for fx in (0.4, 0.5, 0.6) for fy in (0.4, 0.5, 0.6)]
    count = {s: len(page.request.get(f"{api}/cores/{core_id}/{stain}/region?x={s[0]}&y={s[1]}&size=1600").json()["objects"]) for s in spots}
    x, y = max(spots, key=lambda s: count[s])
    return f"/cores/{core_id}?stain={stain}&x={x}&y={y}"


def annotate_target(page: Page, api: str, session: str) -> str:
    fields = page.request.get(f"{api}/annotate/{session}").json()["fields"]
    open_ = [f["tile_id"] for f in fields if f["prepared"] and not f["done_by_me"]]
    if not open_:
        raise SystemExit(f"the annotator has no unfinished field in {session}")
    mine = {t: len(page.request.get(f"{api}/annotate/{session}/fields/{t}").json()["mine"]) for t in open_}
    return f"/annotate/{session}/{max(open_, key=lambda t: mine[t])}"


def select_outline(page: Page) -> None:
    """Zoom in, switch to the outline tool and select a large outlined object near the centre, as a person fixing an
    outline would."""
    page.keyboard.press("o")
    card = page.locator(".card", has=page.locator(".viewer")).first
    card.locator("button", has_text="+").first.click()
    time.sleep(0.8)
    view = page.locator(".viewer").first.bounding_box()
    cx, cy = view["x"] + view["width"] / 2, view["y"] + view["height"] * 0.4
    best, score = None, 0.0
    for polygon in page.locator(".viewer svg polygon").all():
        box = polygon.bounding_box()
        if not box:
            continue
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        if abs(x - cx) < view["width"] * 0.2 and abs(y - cy) < view["height"] * 0.2 and box["width"] * box["height"] > score:
            best, score = (x, y), box["width"] * box["height"]
    if best:
        page.mouse.click(*best)
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base_url", help="where stainID runs, e.g. http://127.0.0.1:18555")
    parser.add_argument("out_dir", type=Path, help="folder for the images (docs/images for the README)")
    parser.add_argument("--password", default="admin", help="admin password (default: admin)")
    parser.add_argument("--replace-default-password", action="store_true",
                        help="if the admin password is still the default, set a random one first so the pages show no default-password "
                             "warning (only for a server started with a throwaway HOME)")
    parser.add_argument("--annotator", default="Ann", help="annotator name to label as (the demo project's Ann has an unfinished field)")
    parser.add_argument("--session", default="neun_review", help="labelling session shown on the annotation page")
    parser.add_argument("--stain", default="AT8", help="stain shown in the core viewer")
    parser.add_argument("--root-label", default="~/stainID/demo_project", help="shown instead of the project folder's absolute path")
    parser.add_argument("--only", nargs="*", default=None, help="file names (or stems) to take, default all")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [n for n in SHOTS if args.only is None or n in args.only or Path(n).stem in args.only]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1500, "height": 911}, color_scheme="light", device_scale_factor=1)
        context.add_init_script("localStorage.setItem('stainid.theme', JSON.stringify('light'))")
        request = context.request
        signed = request.post(f"{base}/api/session/admin", data={"password": args.password})
        if not signed.ok:
            raise SystemExit(f"admin sign-in failed: {signed.text()}")
        if args.replace_default_password and request.get(f"{base}/api/session").json()["default_password"]:
            request.put(f"{base}/api/session/admin/password", data={"current": args.password, "new": secrets.token_urlsafe(12)})
        request.post(f"{base}/api/session/annotator", data={"name": args.annotator})
        project = request.get(f"{base}/api/project").json()  # the project the server was started with (opened last)
        root, api, pages = project["root"], f"{base}/api/projects/{project['id']}", f"{base}/p/{project['id']}"
        page = context.new_page()
        for name in wanted:
            path, width, height = SHOTS[name]
            if name == "viewer.jpg":
                path = viewer_target(page, api, args.stain)
            if name == "annotate.jpg":
                path = annotate_target(page, api, args.session)
            page.set_viewport_size({"width": width, "height": height})
            page.goto(f"{base}/" if name == "projects.png" else pages + path)
            settle(page, 1.5)
            if name == "annotate.jpg":
                select_outline(page)
            page.evaluate(REPLACE_TEXT, [root, args.root_label])
            page.evaluate(REPLACE_TEXT, [str(Path(root).parent), str(Path(args.root_label).parent)])  # the other studies beside it
            target = args.out_dir / name
            if target.suffix == ".jpg":
                page.screenshot(path=str(target), type="jpeg", quality=88)
            else:
                page.screenshot(path=str(target))
            print(f"{target}  ({path})", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
