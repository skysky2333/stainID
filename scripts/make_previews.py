#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict

import cv2

from .utils import discover_images, parse_image_type, ensure_dir, read_image_rgb


def main():
    p = argparse.ArgumentParser(description="Generate downsampled JPEG previews for calibration.")
    p.add_argument("--input_dir", default="origianl_data/out_seg/matched_output", help="Input image directory")
    p.add_argument("--out_root", default="results_stainID", help="Output root for previews")
    p.add_argument("--max_per_type", type=int, default=6, help="Max images per type to preview")
    p.add_argument("--longest_side", type=int, default=2048, help="Preview longest side in pixels")
    args = p.parse_args()

    images = discover_images(args.input_dir)
    grouped = defaultdict(list)
    for pth in images:
        t = parse_image_type(os.path.basename(pth))
        grouped[t].append(pth)

    for t, lst in grouped.items():
        preview_dir = os.path.join(args.out_root, "previews", t, "source")
        ensure_dir(preview_dir)
        for i, pth in enumerate(sorted(lst)[: args.max_per_type]):
            base = os.path.splitext(os.path.basename(pth))[0]
            outp = os.path.join(preview_dir, f"{base}_preview.jpg")
            print(f"Creating preview: {outp}")
            img = read_image_rgb(pth)
            h, w = img.shape[:2]
            if max(h, w) > args.longest_side:
                if h >= w:
                    new_h = args.longest_side
                    new_w = int(w * (args.longest_side / h))
                else:
                    new_w = args.longest_side
                    new_h = int(h * (args.longest_side / w))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(outp, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == "__main__":
    main()

