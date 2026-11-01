"""Repair of the single-component centroid bug in TotalSegmentator 2.18.0 (postprocessing.py).

``ndimage.center_of_mass`` already returns a list for an index array, so wrapping the one-component result in
another list breaks the vertebra post-processing. Run ``python -m common.totalsegmentator_patch`` in env-seg
once; ``check_patch`` verifies the repaired file's hash before every segmentation run.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

from common.provenance import file_hash

ORIGINAL = ("    centers = ndimage.center_of_mass(keep_mask, component_map, keep_components)\n"
            "    if len(keep_components) == 1:\n"
            "        centers = [centers]\n")
REPAIRED = "    centers = ndimage.center_of_mass(keep_mask, component_map, keep_components)\n"


def postprocessing_path():
    if importlib.metadata.version("TotalSegmentator") != "2.18.0":
        raise RuntimeError("This pipeline requires TotalSegmentator 2.18.0 (env-seg)")
    return Path(importlib.util.find_spec("totalsegmentator").origin).parent / "postprocessing.py"


def check_patch():
    path = postprocessing_path()
    text = path.read_text(encoding="utf-8")
    if ORIGINAL in text or text.count(REPAIRED) != 1:
        raise RuntimeError("Run python -m common.totalsegmentator_patch in env-seg before inference")
    return file_hash(path)


def main():
    path = postprocessing_path()
    text = path.read_text(encoding="utf-8")
    if text.count(ORIGINAL) == 1:
        # An index array always produces a list, including the one-component case.
        path.write_text(text.replace(ORIGINAL, REPAIRED), encoding="utf-8", newline="\n")
    print("Verified postprocessing SHA256:", check_patch())


if __name__ == "__main__":
    main()
