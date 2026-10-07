"""Exact 90-degree rotations of an annotation (image + every box), for the orientation layer.

Orientation is the cheapest layer to annotate (one integer per image) and the cheapest to
*multiply*: rotating an upright, fully annotated page by 90/180/270 yields three more images whose
every box is still exact. So the annotator records the TRUE capture orientation once, and the
orientation-training examples are generated here instead of hand-labelled.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .schema import DocAnnotation

# PIL transpose op that rotates the image CLOCKWISE by the key (degrees)
_PIL_CW = {90: "ROTATE_270", 180: "ROTATE_180", 270: "ROTATE_90"}


def _rot_point(x: float, y: float, W: float, H: float, deg: int) -> tuple[float, float]:
    if deg == 90:
        return H - y, x
    if deg == 180:
        return W - x, H - y
    if deg == 270:
        return y, W - x
    return x, y


def rotate_box(b: list[float], W: float, H: float, deg: int) -> list[float]:
    """Box in a W x H image -> the same box after rotating the image ``deg`` degrees clockwise."""
    (ax, ay), (bx, by) = _rot_point(b[0], b[1], W, H, deg), _rot_point(b[2], b[3], W, H, deg)
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]


def rotate_annotation(ann: DocAnnotation, deg: int, image: str | None = None) -> DocAnnotation:
    """Return a copy of ``ann`` for the stored image rotated ``deg`` (90/180/270) clockwise.

    ``page.orientation`` accumulates (an upright page rotated 90 has orientation 90; rotating a
    page with orientation 90 by 270 brings it back upright). ``image`` replaces the image path."""
    deg %= 360
    if deg not in (0, 90, 180, 270):
        raise ValueError(f"only multiples of 90 are exact, got {deg}")
    out = copy.deepcopy(ann)
    W, H = ann.page.width, ann.page.height
    for coll in (out.layout, out.ocr):
        for o in coll:
            o.bbox = rotate_box(o.bbox, W, H, deg)
    for ln in out.ocr:
        if ln.poly:
            ln.poly = [list(_rot_point(x, y, W, H, deg)) for x, y in ln.poly]
    if deg in (90, 270):
        out.page.width, out.page.height = H, W
    out.page.orientation = (ann.page.orientation + deg) % 360
    out.doc_id = f"{ann.doc_id}@rot{out.page.orientation}" if deg else ann.doc_id
    if image is not None:
        out.image = image
    return out


def to_upright(ann: DocAnnotation, image: str | None = None) -> DocAnnotation:
    """Rotate a record (as captured) so that ``page.orientation`` becomes 0."""
    return rotate_annotation(ann, (360 - ann.page.orientation) % 360, image=image)


def materialize_rotations(ann: DocAnnotation, image_path: str, out_dir: str,
                          angles=(90, 180, 270)) -> list[tuple[DocAnnotation, str]]:
    """Write rotated copies of the image and return ``(annotation, image_path)`` pairs.

    Paths in the returned annotations are absolute so exported samples are self-contained."""
    from PIL import Image
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    pairs = []
    with Image.open(image_path) as im:
        im.load()
        for deg in angles:
            deg %= 360
            if deg == 0:
                continue
            rot = im.transpose(getattr(Image.Transpose, _PIL_CW[deg]))
            dst = out_dir_p / f"{ann.doc_id.replace('/', '_')}_rot{deg}.png"
            rot.save(dst)
            pairs.append((rotate_annotation(ann, deg, image=str(dst.resolve())), str(dst.resolve())))
    return pairs
