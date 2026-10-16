"""Stage 3, outlines: Segment Anything's three outputs (small, medium, large) for each candidate, prompted with one point
at the candidate as the one-click tool does, with SAM's predicted IoU and stability score, and the default among them.

The image encoder runs once per 1024-px window of a field (fp16 on the GPU, as `stainid.masks.clicks`; fp32 on the CPU
without one) and the prompts of every candidate in that window are decoded in batches. An output that runs away (more
than `runaway_factor` × its candidate's area), touches the window edge, or falls outside the stain's size limits is not
a valid default; the default is the output the stain's outline step uses (NeuN large, 6E10 medium, AT8 the one with the
best DAB contrast) when valid, else the valid one with the best predicted IoU. With no valid output the candidate keeps
its own outline (`sam_status` fallback_runaway / fallback_window_edge / fallback_size). All three outputs are stored
either way, so annotation can switch between them without calling SAM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from stainid import candidates as store
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.core_images import open_image
from stainid.masks import clicks
from stainid.masks.objects import AREA_LIMITS_UM2, MASK_RULES, ring_contrast, shape_features
from stainid.masks.sam import window_origin
from stainid.pipelines.cohort_v1 import context_crop
from stainid.v2 import manifest
from stainid.v2.params import params, stage_params

Image.MAX_IMAGE_PIXELS = None
BATCH = 32
COLUMNS = ["candidate_id", "sam_status", "sam_default", "sam_outline", "sam_masks", "sam_scores", "sam_area_um2", "sam_perimeter_um",
           "sam_equivalent_diameter_um", "sam_major_axis_um", "sam_minor_axis_um", "sam_elongation", "sam_eccentricity", "sam_solidity",
           "sam_circularity", "sam_mean_dab_od", "sam_p90_dab_od", "sam_normalized_mean_dab", "sam_ring_contrast", "sam_core_dab_q"]


class BatchOutliner:
    """Segment Anything for many prompts: encoder fp16 on the GPU (fp32 on the CPU), prompts decoded in batches."""

    def __init__(self, checkpoint, device: str | None = None):
        import torch

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = clicks.build_model(checkpoint)
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.image_encoder.half()
        self.model.to(self.device).eval()
        from segment_anything.utils.transforms import ResizeLongestSide

        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def encode(self, window: np.ndarray):
        torch = self.torch
        transformed = self.transform.apply_image(window)
        x = torch.as_tensor(transformed, device=self.device).permute(2, 0, 1).contiguous()[None].float()
        x = self.model.preprocess(x)
        with torch.inference_mode():
            features = self.model.image_encoder(x.half() if self.half else x).float()
        return features, tuple(window.shape[:2]), tuple(transformed.shape[:2])

    def decode(self, embedding, points: np.ndarray) -> list[list["Output"]]:
        """For (n, 2) window points: per point SAM's three outputs, each as its area, whether it touches the window edge,
        its box in the window and the mask cropped to that box (a pixel of margin), with the predicted IoU and stability
        score. Only the crops leave the GPU."""
        from segment_anything.utils.amg import calculate_stability_score

        torch = self.torch
        features, original, input_size = embedding
        out: list[list[Output]] = []
        for start in range(0, len(points), BATCH):
            chunk = points[start: start + BATCH]
            coords = self.transform.apply_coords(chunk.astype(float), original)
            coords_t = torch.as_tensor(coords, dtype=torch.float32, device=self.device)[:, None, :]
            labels_t = torch.ones((len(chunk), 1), dtype=torch.int64, device=self.device)
            with torch.inference_mode():
                sparse, dense = self.model.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
                low, iou = self.model.mask_decoder(image_embeddings=features, image_pe=self.model.prompt_encoder.get_dense_pe(),
                                                   sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
                logits = self.model.postprocess_masks(low, input_size, original)
                stability = calculate_stability_score(logits, self.model.mask_threshold, 1.0).float().cpu().numpy()
                masks = logits > self.model.mask_threshold
                area = masks.sum(dim=(2, 3)).cpu().numpy()
                edge = (masks[:, :, 0, :].any(-1) | masks[:, :, -1, :].any(-1) | masks[:, :, :, 0].any(-1) | masks[:, :, :, -1].any(-1)).cpu().numpy()
                rows, cols = masks.any(dim=3).cpu().numpy(), masks.any(dim=2).cpu().numpy()
                iou = iou.float().cpu().numpy()
                for b in range(len(chunk)):
                    outputs = []
                    for k in range(masks.shape[1]):
                        ys, xs = np.flatnonzero(rows[b, k]), np.flatnonzero(cols[b, k])
                        if not len(ys):
                            outputs.append(Output(0, False, (0, 0, 0, 0), np.zeros((0, 0), bool), float(iou[b, k]), float(stability[b, k])))
                            continue
                        y0, y1 = max(0, ys[0] - 1), min(masks.shape[2], ys[-1] + 2)
                        x0, x1 = max(0, xs[0] - 1), min(masks.shape[3], xs[-1] + 2)
                        outputs.append(Output(int(area[b, k]), bool(edge[b, k]), (int(x0), int(y0), int(x1), int(y1)),
                                              masks[b, k, y0:y1, x0:x1].cpu().numpy(), float(iou[b, k]), float(stability[b, k])))
                    out.append(outputs)
        return out


@dataclass
class Output:
    """One of SAM's outputs for a prompt: area (px), touches the window edge, box (x0, y0, x1, y1) in the window, the mask
    in that box, predicted IoU and stability score."""
    area: int
    edge: bool
    box: tuple[int, int, int, int]
    mask: np.ndarray
    iou: float
    stability: float


def _ring_contrast(output: Output, dab: np.ndarray, ring_px: int = 4) -> float:
    """`stainid.masks.objects.ring_contrast` of an output, from its box grown by the ring (the window's DAB)."""
    x0, y0, x1, y1 = output.box
    gx0, gy0 = max(0, x0 - ring_px - 1), max(0, y0 - ring_px - 1)
    gx1, gy1 = min(dab.shape[1], x1 + ring_px + 1), min(dab.shape[0], y1 + ring_px + 1)
    mask = np.zeros((gy1 - gy0, gx1 - gx0), bool)
    mask[y0 - gy0: y1 - gy0, x0 - gx0: x1 - gx0] = output.mask
    return ring_contrast(mask, dab[gy0:gy1, gx0:gx1], ring_px)


def choose(outputs: list[Output], candidate_px: float, stain: str, pixel_size_um: float, dab: np.ndarray,
           runaway_factor: float) -> tuple[int, str]:
    """The default output among a candidate's three and the status ("ok" or the fallback reason)."""
    lower, upper = AREA_LIMITS_UM2[stain]
    reasons = []
    for output in outputs:
        if output.area > runaway_factor * max(candidate_px, 1):
            reasons.append("fallback_runaway")
        elif output.edge and output.area:
            reasons.append("fallback_window_edge")
        elif not lower <= output.area * pixel_size_um ** 2 <= upper:
            reasons.append("fallback_size")
        else:
            reasons.append("")
    valid = [i for i, r in enumerate(reasons) if not r]
    if not valid:
        order = ("fallback_runaway", "fallback_window_edge", "fallback_size")
        return -1, min(reasons, key=order.index)
    rule = MASK_RULES[stain]
    if rule == "contrast":
        index = max((i for i in valid if i in (1, 2) and outputs[i].area >= 20), key=lambda i: _ring_contrast(outputs[i], dab), default=None)
    else:
        index = rule if rule in valid else None
    if index is None:
        index = max(valid, key=lambda i: outputs[i].iou)
    return int(index), "ok"


def _polygon(output: Output, origin: tuple[int, int]) -> list[int]:
    """The outline of an output's largest piece, as `stainid.masks.clicks.polygon` makes it (holes do not change an outer
    contour, so they are not filled first)."""
    if output.area < clicks.MIN_PIXELS:
        return []
    contours, _ = cv2.findContours(np.pad(output.mask, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1:
        return []
    contour = cv2.approxPolyDP(contour, clicks.EPSILON_PX, True)[:, 0, :]
    if len(contour) < 3:
        return []
    return (contour + np.array([origin[0] + output.box[0] - 1, origin[1] + output.box[1] - 1])).ravel().tolist()


def outline_field(outliner: BatchOutliner, rgb: np.ndarray, field_origin: tuple[int, int], stain: str, threshold: float, pixel_size_um: float,
                  rows: pd.DataFrame, runaway_factor: float) -> list[dict]:
    """The three outputs and the default of every candidate row (core coordinates) in one field crop."""
    out = []
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    smooth = cv2.GaussianBlur(dab, (0, 0), 1.0)
    fx, fy = field_origin
    xs, ys = rows.x_core.to_numpy(float) - fx, rows.y_core.to_numpy(float) - fy
    windows = [window_origin(rgb.shape[:2], x, y) for x, y in zip(xs, ys)]
    for origin in sorted(set(windows)):
        index = [i for i, w in enumerate(windows) if w == origin]
        x0, y0 = origin
        window = rgb[y0: y0 + 1024, x0: x0 + 1024]
        local_dab = dab[y0: y0 + window.shape[0], x0: x0 + window.shape[1]]
        decoded = outliner.decode(outliner.encode(window), np.column_stack((xs[index] - x0, ys[index] - y0)))
        for i, outputs in zip(index, decoded):
            row = rows.iloc[i]
            chosen, status = choose(outputs, float(row.area_um2) / pixel_size_um ** 2, stain, pixel_size_um, local_dab, runaway_factor)
            polygons = [_polygon(o, (fx + x0, fy + y0)) for o in outputs]
            order = np.argsort([o.area for o in outputs], kind="stable")  # small → large
            result = {"candidate_id": row.candidate_id, "sam_status": status,
                      "sam_default": int(np.flatnonzero(order == chosen)[0]) if chosen >= 0 else -1,
                      "sam_masks": ";".join(store.encode_points(polygons[j]) for j in order),
                      "sam_scores": ";".join(f"{outputs[j].iou:.4f} {outputs[j].stability:.4f}" for j in order),
                      "sam_outline": store.encode_points(polygons[chosen]) if chosen >= 0 else ""}
            if chosen >= 0:
                best = outputs[chosen]
                bx0, by0, bx1, by1 = best.box
                pad = 8
                gx0, gy0 = max(0, bx0 - pad), max(0, by0 - pad)
                gx1, gy1 = min(local_dab.shape[1], bx1 + pad), min(local_dab.shape[0], by1 + pad)
                crop = np.zeros((gy1 - gy0, gx1 - gx0), bool)
                crop[by0 - gy0: by1 - gy0, bx0 - gx0: bx1 - gx0] = best.mask
                features = shape_features(crop, local_dab[gy0:gy1, gx0:gx1], pixel_size_um, threshold)
                result |= {f"sam_{k_}": v for k_, v in features.items()}
                values = smooth[y0 + gy0: y0 + gy1, x0 + gx0: x0 + gx1][crop] / threshold
                result["sam_core_dab_q"] = " ".join(f"{v:.4f}" for v in np.quantile(values, np.linspace(0, 1, 21)))
            out.append(result)
    return out


def selected(frame: pd.DataFrame, calls: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The candidates this stage outlines: all, the accepted ones, or (probable) accepted or probability ≥ min_probability."""
    if settings["select"] == "all" or calls.empty:
        return frame
    merged = frame.merge(calls[["candidate_id", "decision", "probability"]], on="candidate_id", how="left")
    keep = merged.decision.eq("accepted")
    if settings["select"] == "probable":
        keep |= pd.to_numeric(merged.probability, errors="coerce").fillna(0) >= float(settings["min_probability"])
    return frame[keep.to_numpy()]


def expected(project, stain: str, core_id: str) -> dict | None:
    upstream = manifest.read(project, "proposals", stain, core_id)
    if upstream is None:
        return None
    calls = manifest.read(project, "calls", stain, core_id)
    settings = stage_params(project, "outlines", stain)
    return manifest.make("outlines", settings, {"proposals": upstream["digest"], "calls": calls["digest"] if calls and settings["select"] != "all" else "",
                                                "sam": manifest.file_signature(project.model("sam"))})


def outline_core(project, stain: str, core_id: str, tiles: list[dict], outliner: BatchOutliner, threshold: float, redo: bool = False) -> dict:
    """Outline the selected candidates of one core image; candidates outlined before (same proposals, same SAM) are kept
    unless `redo`."""
    started = time.time()
    settings = params(project)["outlines"]
    frame = store.read_table(store.path(project, stain, core_id, "proposals"))
    calls = store.read_table(store.path(project, stain, core_id, "calls"))
    target = store.root(project) / "outlines" / stain / f"{core_id}.csv.gz"
    previous = store.read_table(target)
    recorded = manifest.read(project, "outlines", stain, core_id)
    made = expected(project, stain, core_id)
    same_base = recorded is not None and recorded["inputs"].get("proposals") == made["inputs"]["proposals"] and \
        recorded["inputs"].get("sam") == made["inputs"]["sam"] and recorded.get("params", {}).get("runaway_factor") == settings["runaway_factor"]
    done = set(previous.candidate_id) if same_base and not redo and not previous.empty else set()
    wanted = selected(frame, calls, settings)
    todo = wanted[~wanted.candidate_id.isin(done)]
    results = []
    if not todo.empty:
        by_tile = {t["tile_id"]: t for t in tiles}
        with open_image(project.root / tiles[0]["image_path"]) as image:
            image.load()
            for tile_id, rows in todo.groupby("tile_id", sort=True):
                tile = by_tile[tile_id]
                x, y = int(tile["x_px"]), int(tile["y_px"])
                rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), project.context_px)
                pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
                results += outline_field(outliner, rgb, (x - inner[1].start, y - inner[0].start), stain, threshold, pixel_size, rows,
                                         float(settings["runaway_factor"]))
    kept = previous if done else previous.iloc[0:0]
    table = pd.concat([kept, pd.DataFrame(results, columns=COLUMNS)], ignore_index=True) if results else kept.reindex(columns=COLUMNS)
    store.write_table(target, table.reindex(columns=COLUMNS))
    manifest.write(project, "outlines", stain, core_id, made, outlined=len(table), new=len(results), seconds=round(time.time() - started, 1))
    from stainid.v2.classify import assemble

    assemble(project, stain, core_id)
    return {"outlined": len(table), "new": len(results), "seconds": time.time() - started}
