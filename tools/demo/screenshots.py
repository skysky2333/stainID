"""Take the README screenshots of a running stainID web app on the synthetic demo project (never on study data).

    python tools/demo/screenshots.py http://127.0.0.1:18555 docs/images [--only workflow annotate]

Needs Playwright (`pip install playwright && python -m playwright install chromium`, best in a separate virtual
environment; stainID itself does not need it). Start the app first on a project made by make_demo_project.py, with a
throwaway HOME, so that neither the list of recent projects nor the admin password of your own stainID settings
changes:

    HOME=/tmp/stainid-demo-home stainid --project /tmp/demo_project serve --no-browser --host 127.0.0.1 --port 18555

(add `--projects-dir` with a folder holding the demo project and a project made by make_setup_demo.py for a Projects
page with more than one study). The script refuses a project that is not the synthetic demo (it reads the demo's
demo.json). It signs in as admin, opens each page of the project the server was started with (at /p/<project>/...) in
the light theme and writes:

- projects.png, home.png, workflow.png (the Run page with its flow chart), models.png, labels.png, results.png,
  settings.png;
- viewer.jpg: the Cores view of the AD core with the most tau+ neurons near its centre;
- annotate.jpg: "Fix the field" on the field the demo's Ann has started (her changes: classes in their colours, removed
  and rejected candidates dotted grey), zoomed in a little, pointing at an object (its name beside it, its facts in the
  side panel);
- annotate-precise.jpg: the same field in precise editing (the dots to drag);
- tutorial.jpg: the tutorial a new annotator (Dana) is sent to first.
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

SHOTS = {  # name: (path in the project, or None when chosen from the data, viewport width, height)
    "projects.png": ("", 1400, 640),  # the Projects page (the server's, not a project's)
    "workflow.png": ("/workflow", 1500, 1000),
    "home.png": ("/", 1500, 911),
    "models.png": ("/models", 1500, 911),
    "viewer.jpg": (None, 1500, 1000),
    "labels.png": ("/labels", 1400, 900),
    "annotate.jpg": (None, 1500, 900),
    "annotate-precise.jpg": (None, 1500, 900),
    "tutorial.jpg": ("/annotate", 1500, 900),
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
    found = page.request.get(f"{api}/cores/{core_id}/{stain}/objects").json()
    xy = [(r[found["columns"].index("x")], r[found["columns"].index("y")]) for r in found["rows"]]
    count = {s: sum(abs(x - s[0]) < 800 and abs(y - s[1]) < 800 for x, y in xy) for s in spots}
    x, y = max(spots, key=lambda s: count[s])
    return f"/cores/{core_id}?stain={stain}&x={x}&y={y}"


def point_at(page: Page, selector: str, near: tuple[float, float] | None = None) -> None:
    """Move the mouse over the middle of the outline (of these) closest to `near` (the picture's centre by default)."""
    view = page.locator(".viewer").first.bounding_box()
    cx, cy = near or (view["x"] + view["width"] * 0.5, view["y"] + view["height"] * 0.5)
    best, distance = None, float("inf")
    for shape in page.locator(selector).all():
        box = shape.bounding_box()
        if not box or box["width"] < 8:
            continue
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        d = (x - cx) ** 2 + (y - cy) ** 2
        if view["x"] < x < view["x"] + view["width"] and view["y"] < y < view["y"] + view["height"] and d < distance:
            best, distance = (x, y), d
    if best:
        page.mouse.move(*best)
        time.sleep(0.6)


def zoom_in(page: Page, steps: int = 3) -> None:
    view = page.locator(".viewer").first.bounding_box()
    page.mouse.move(view["x"] + view["width"] * 0.5, view["y"] + view["height"] * 0.55)
    for _ in range(steps):
        page.mouse.wheel(0, -120)
        time.sleep(0.15)
    time.sleep(0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base_url", help="where stainID runs, e.g. http://127.0.0.1:18555")
    parser.add_argument("out_dir", type=Path, help="folder for the images (docs/images for the README)")
    parser.add_argument("--password", default="admin", help="admin password (default: admin)")
    parser.add_argument("--replace-default-password", action="store_true",
                        help="if the admin password is still the default, set a random one first so the pages show no default-password "
                             "warning (only for a server started with a throwaway HOME)")
    parser.add_argument("--annotator", default="Ann", help="annotator name to label as (the demo project's Ann has an unfinished field)")
    parser.add_argument("--newcomer", default="Dana", help="a name that has not been through the tutorial yet")
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
        project = request.get(f"{base}/api/project").json()  # the project the server was started with (opened last)
        root, api, pages = project["root"], f"{base}/api/projects/{project['id']}", f"{base}/p/{project['id']}"
        demo = Path(root) / "demo.json"
        if not demo.exists():
            raise SystemExit(f"{root} is not the synthetic demo (no demo.json): screenshots are only taken of the demo")
        rounds = json.loads(demo.read_text())["rounds"]
        page = context.new_page()
        for name in wanted:
            path, width, height = SHOTS[name]
            person = args.newcomer if name == "tutorial.jpg" else args.annotator
            request.post(f"{base}/api/session/annotator", data={"name": person})
            if name == "viewer.jpg":
                path = viewer_target(page, api, args.stain)
            if name in ("annotate.jpg", "annotate-precise.jpg"):
                path = f"/annotate/{rounds['fix']}/{rounds['ann_field']}"
            page.set_viewport_size({"width": width, "height": height})
            page.goto(f"{base}/" if name == "projects.png" else pages + path)
            settle(page, 1.5)
            if name == "annotate.jpg":
                zoom_in(page, 3)
                point_at(page, ".round-object.look-class polygon:last-child")
            if name == "annotate-precise.jpg":
                zoom_in(page, 2)
                point_at(page, ".round-object.look-class polygon:last-child")
                page.mouse.down()  # a click on the object: precise editing
                page.mouse.up()
                time.sleep(1.2)
            if name == "workflow.png":
                page.locator(".flow-node", has_text="Find candidates").first.hover()
                time.sleep(0.5)
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
