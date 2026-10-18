"""Take the README screenshots of a running stainID web app on the synthetic demo project (never on study data).

    python tools/demo/screenshots.py http://127.0.0.1:18555 docs/images [--only workflow annotate]

Needs Playwright (`pip install playwright && python -m playwright install chromium`, best in a separate virtual
environment; stainID itself does not need it). Start the app first on a project made by make_demo_project.py, with a
throwaway HOME, so that neither the list of recent projects nor the admin password of your own stainID settings
changes:

    HOME=/tmp/stainid-demo-home stainid --project /tmp/demo_project serve --no-browser --host 127.0.0.1 --port 18555

(add `--projects-dir` with a folder holding the demo project and a project made by make_setup_demo.py for a Projects
page with more than one study). The script refuses a project that is not the synthetic demo (it reads the demo's
demo.json). It signs in as admin, makes accounts for the demo's annotator and a newcomer (members of the demo's
sessions; random passwords, kept nowhere), opens each page of the project the server was started with (at
/p/<project>/...) in the light theme, the annotation pages signed in as those people, and writes:

- projects.png, home.png, workflow.png (the Run page with its flow chart), models.png, labels.png, results.png,
  settings.png;
- viewer.jpg: the Cores view of the AD core with the most tau+ neurons near its centre;
- viewer-annotations.jpg: the Cores view of the core of Ann's field with the Annotations layer on (where people's labels
  differ from detection), the excluded areas shaded and the minimap with each tile's review status;
- annotate.jpg: "Fix the field" on the field the demo's Ann has started (her changes: classes in their colours, removed
  and rejected candidates dotted grey; the acceptance-threshold slider at the side), zoomed in a little, pointing at an
  object (its name beside it, its facts in the side panel);
- annotate-precise.jpg: the same field in precise editing (the dots to drag);
- annotate-brush.jpg: the same field with the brush on (B) and one stroke of a bad area painted (not saved);
- annotate-split.jpg / annotate-merge.jpg: splitting an object in two (S, the preview before Enter) and two objects
  chosen to merge (M) (neither saved);
- areas.jpg: a "Mark bad areas" core, every stain image of it as a tab, with a stroke painted (not saved);
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
    "viewer-annotations.jpg": (None, 1500, 1000),
    "labels.png": ("/labels", 1400, 900),
    "annotate.jpg": (None, 1500, 900),
    "annotate-precise.jpg": (None, 1500, 900),
    "annotate-brush.jpg": (None, 1500, 900),
    "annotate-split.jpg": (None, 1500, 900),
    "annotate-merge.jpg": (None, 1500, 900),
    "areas.jpg": (None, 1500, 950),
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


def objects_near_centre(page: Page, n: int) -> list[tuple[float, float, float]]:
    """The middles and sizes of the `n` drawn objects closest to the picture's centre (screen px)."""
    view = page.locator(".viewer").first.bounding_box()
    cx, cy = view["x"] + view["width"] * 0.5, view["y"] + view["height"] * 0.5
    found = []
    for shape in page.locator(".round-object.look-class polygon:last-child").all():
        box = shape.bounding_box()
        if box and box["width"] >= 8:
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            if view["x"] < x < view["x"] + view["width"] and view["y"] < y < view["y"] + view["height"]:
                found.append((x, y, box["width"] * box["height"]))
    return sorted(found, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)[:n]


def stroke(page: Page, selector: str, fx: float = 0.3, fy: float = 0.3, length: float = 0.25) -> None:
    """One brush stroke across part of the picture (a drag), left to right with a little wave."""
    view = page.locator(selector).first.bounding_box()
    x, y = view["x"] + view["width"] * fx, view["y"] + view["height"] * fy
    page.mouse.move(x, y)
    page.mouse.down()
    for k in range(30):
        page.mouse.move(x + view["width"] * length * k / 29, y + 18 * ((k % 10) / 9))
    page.mouse.up()
    time.sleep(0.5)


def zoom_in_at(page: Page, x: float, y: float, steps: int) -> None:
    """Zoom in with the wheel at this screen point, then pan it to the picture's middle (middle-button drag)."""
    page.mouse.move(x, y)
    for _ in range(steps):
        page.mouse.wheel(0, -120)
        time.sleep(0.15)
    time.sleep(0.5)
    view = page.locator(".viewer").first.bounding_box()
    page.mouse.down(button="middle")
    page.mouse.move(view["x"] + view["width"] * 0.5, view["y"] + view["height"] * 0.5, steps=8)
    page.mouse.up(button="middle")
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

        def new_context():
            made = browser.new_context(viewport={"width": 1500, "height": 911}, color_scheme="light", device_scale_factor=1)
            made.add_init_script("localStorage.setItem('stainid.theme', JSON.stringify('light'))")
            made.add_init_script(f"localStorage.setItem('stainid.tutorial.areas.{args.annotator}', '1')")  # Ann knows "Mark bad areas"
            return made

        context = new_context()
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
        admin_page = context.new_page()
        signed_in: dict[str, Page] = {}

        def as_person(person: str) -> Page:
            """A browser of this person (an annotator account made here, a member of the demo's sessions)."""
            if person in signed_in:
                return signed_in[person]
            password = secrets.token_urlsafe(12)
            made = request.post(f"{base}/api/session/people", data={"name": person, "password": password})
            reset = None
            if made.status == 409:  # made by an earlier run: a one-time password, then a new one
                found = next(a for a in request.get(f"{base}/api/session/people").json()["people"] if a["name"] == person)
                reset = request.post(f"{base}/api/session/people/{found['id']}/reset").json()["password"]
            elif not made.ok:
                raise SystemExit(f"could not make an account for {person}: {made.text()}")
            browser_of = new_context()
            signed = browser_of.request.post(f"{base}/api/session/login", data={"name": person, "password": reset or password})
            if not signed.ok:
                raise SystemExit(f"{person} could not sign in: {signed.text()}")
            if reset:
                browser_of.request.put(f"{base}/api/session/password", data={"current": reset, "new": password})
            account = signed.json()["account"]["id"]
            for session in {rounds.get("fix"), rounds.get("areas")} - {None}:
                request.post(f"{api}/labels/sessions/{session}/access/members", data={"account": account})
            signed_in[person] = browser_of.new_page()
            return signed_in[person]

        for name in wanted:
            path, width, height = SHOTS[name]
            if name == "tutorial.jpg":
                page = as_person(args.newcomer)
            elif name.startswith("annotate") or name == "areas.jpg":
                page = as_person(args.annotator)
            else:
                page = admin_page
            if name == "viewer.jpg":
                path = viewer_target(page, api, args.stain)
            if name.startswith("annotate"):
                path = f"/annotate/{rounds['fix']}/{rounds['ann_field']}"
            if name == "viewer-annotations.jpg":
                path = f"/cores/{rounds['ann_field'].rsplit('_', 2)[0]}?stain={rounds['ann_field'].rsplit('_', 2)[1]}"
            if name == "areas.jpg":
                if not rounds.get("areas"):
                    print("areas.jpg: the demo has no Mark bad areas session (make the demo again)", flush=True)
                    continue
                path = f"/annotate/{rounds['areas']}"
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
            if name == "annotate-brush.jpg":
                zoom_in(page, 1)
                page.keyboard.press("b")
                for _ in range(4):
                    page.keyboard.press("Shift+Tab")  # a bigger brush
                stroke(page, ".annotate-image .viewer", 0.25, 0.3, 0.3)
            if name == "annotate-split.jpg":
                zoom_in(page, 3)
                near = objects_near_centre(page, 8)
                if near:
                    x, y, _ = max(near, key=lambda q: q[2])  # the largest object near the middle
                    page.mouse.move(x, y)
                    zoom_in_at(page, x, y, 3)
                    near = objects_near_centre(page, 1)
                    page.keyboard.press("s")
                    page.mouse.click(*near[0][:2])
                    time.sleep(2.5)  # the preview of the two pieces
            if name == "annotate-merge.jpg":
                zoom_in(page, 2)
                near = objects_near_centre(page, 12)
                pairs = [(a, b) for i, a in enumerate(near) for b in near[i + 1:]]
                if pairs:
                    a, b = min(pairs, key=lambda ab: (ab[0][0] - ab[1][0]) ** 2 + (ab[0][1] - ab[1][1]) ** 2)  # the closest two
                    zoom_in_at(page, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 3)
                    page.keyboard.press("m")
                    for x, y, _ in sorted(objects_near_centre(page, 2)):
                        page.mouse.click(x, y)
                        time.sleep(0.4)
            if name == "viewer-annotations.jpg":
                page.locator("label.layer-toggle", has_text="Annotations").first.locator("input").check()
                settle(page, 2.0)
                mini = page.locator("svg.minimap").first.bounding_box()  # to the tiles in progress (Ann's field)
                page.mouse.click(mini["x"] + mini["width"] * 0.62, mini["y"] + mini["height"] * 0.3)
                settle(page, 1.0)
                view = page.locator(".dz").first.bounding_box()
                page.mouse.move(view["x"] + view["width"] * 0.5, view["y"] + view["height"] * 0.5)
                for _ in range(3):
                    page.mouse.wheel(0, -120)
                    time.sleep(0.2)
                settle(page, 2.0)
            if name == "areas.jpg":
                page.wait_for_selector(".area-pending", timeout=120000)
                settle(page, 3.0)
                for _ in range(5):
                    page.keyboard.press("Shift+Tab")
                stroke(page, ".dz", 0.3, 0.35, 0.2)
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
