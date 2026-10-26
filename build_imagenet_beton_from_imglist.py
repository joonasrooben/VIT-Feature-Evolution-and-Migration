# build_imagenet_beton_from_imglist.py
from __future__ import annotations

import argparse
import ast
import os
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


class OpenOODImglistDataset(Dataset):
    """
    Supports OpenOOD-style lines like:
      relative/path.jpg 123
      relative/path.jpg {"label": 123, ...}
    """

    def __init__(self, imglist_path: str, data_root: str):
        self.samples: list[Tuple[str, int]] = []

        with open(imglist_path, "r") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                image_name, extra_str = raw_line.split(" ", 1)
                extra_str = extra_str.strip()

                if extra_str[0] == "{":
                    extra = ast.literal_eval(extra_str)
                    label = int(extra["label"])
                else:
                    label = int(extra_str)

                full_path = os.path.join(data_root, image_name)
                self.samples.append((full_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)

        # FFCV expects the tuple order to match the writer field order
        return arr, label


def main():
    parser = argparse.ArgumentParser(description="Convert OpenOOD imglist split to FFCV .beton")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--imglist", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--write-mode", type=str, default="raw",
                        choices=["raw", "jpg", "proportion", "smart"])
    parser.add_argument("--max-resolution", type=int, default=256,
                        help="Resize images with larger side above this threshold")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--compress-probability", type=float, default=0.5)
    parser.add_argument("--smart-threshold", type=int, default=500 * 500)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    dataset = OpenOODImglistDataset(
        imglist_path=args.imglist,
        data_root=args.data_root,
    )

    image_field_kwargs = {
        "write_mode": args.write_mode,
        "max_resolution": args.max_resolution,
    }

    if args.write_mode != "raw":
        image_field_kwargs["jpeg_quality"] = args.jpeg_quality

    if args.write_mode == "proportion":
        image_field_kwargs["compress_probability"] = args.compress_probability

    if args.write_mode == "smart":
        image_field_kwargs["smart_threshold"] = args.smart_threshold

    writer = DatasetWriter(
        args.output,
        {
            "image": RGBImageField(**image_field_kwargs),
            "label": IntField(),
        },
        num_workers=args.num_workers,
    )

    writer.from_indexed_dataset(dataset)
    print(f"Wrote {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()