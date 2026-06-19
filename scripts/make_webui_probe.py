#!/usr/bin/env python3
"""Web UI/UX understanding probe (for web agents). Renders a synthetic web page and asks the
questions an agent must answer to act: locate interactive elements (button/search/cart),
identify the primary call-to-action, read the nav, and reason about affordances ("what would you
click to check out?"). Element boxes are exact (recorded at draw time) so spotting is gradable.

Output: data/benchmarks/webui_probe/{images,webui.jsonl,sample.*}
    python scripts/make_webui_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.fonts import load_font  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

OUT = ROOT / "data" / "benchmarks" / "webui_probe"
IMG = OUT / "images"
W, H = 1000, 640


def render():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    boxes = {}
    # top nav bar
    d.rectangle([0, 0, W, 60], fill=(28, 40, 64))
    d.text((24, 18), "ACME Shop", fill="white", font=load_font(24, True))
    nav = ["Home", "Products", "Pricing", "About"]
    x = 360
    for item in nav:
        d.text((x, 20), item, fill=(220, 225, 235), font=load_font(20))
        x += 130
    # login button (top-right)
    login = [860, 14, 970, 46]
    d.rectangle(login, outline="white", width=2)
    d.text((878, 20), "Login", fill="white", font=load_font(18))
    boxes["login"] = login
    # search box
    search = [360, 90, 760, 128]
    d.rectangle(search, outline=(150, 150, 150), width=2)
    d.text((372, 98), "Search products...", fill=(150, 150, 150), font=load_font(18))
    boxes["search"] = search
    # hero + primary CTA
    d.text((80, 210), "Summer Sale — up to 50% off", fill=(20, 20, 20), font=load_font(36, True))
    cta = [80, 300, 280, 356]
    d.rounded_rectangle(cta, radius=8, fill=(220, 70, 40))
    d.text((110, 314), "Get Started", fill="white", font=load_font(24, True))
    boxes["cta"] = cta
    # cart / checkout (right)
    cart = [840, 300, 960, 356]
    d.rounded_rectangle(cart, radius=8, fill=(40, 140, 70))
    d.text((858, 314), "Checkout", fill="white", font=load_font(20, True))
    boxes["checkout"] = cart
    # footer
    d.rectangle([0, H - 40, W, H], fill=(240, 240, 240))
    d.text((24, H - 32), "© 2025 ACME · Privacy · Terms", fill=(120, 120, 120), font=load_font(16))
    return im, boxes


def bx(b):
    return f"{b[0]},{b[1]},{b[2]},{b[3]};{W},{H}"


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    im, boxes = render()
    p = str(IMG / "shop.png")
    im.save(p)
    samples = [
        Sample("ui_login_box", p,
               f"Return the bounding box of the Login button as [x1,y1,x2,y2] in pixels. Image {W}x{H}.",
               [bx(boxes["login"])], "webui-locate", "grounding",
               {"content_class": "webui", "task": "locate", "spotting": ",".join(map(str, boxes["login"]))}),
        Sample("ui_cta_text", p, "What is the text on the primary call-to-action button?",
               ["Get Started"], "webui-affordance", "exact",
               {"content_class": "webui", "task": "primary-cta"}),
        Sample("ui_nav_list", p, "List the navigation menu items, comma-separated.",
               ["Home, Products, Pricing, About"], "webui-read", "ned",
               {"content_class": "webui", "task": "read-nav"}),
        Sample("ui_checkout_box", p,
               f"To buy items, which button would you click? Return its bounding box [x1,y1,x2,y2]. Image {W}x{H}.",
               [bx(boxes["checkout"])], "webui-affordance", "grounding",
               {"content_class": "webui", "task": "affordance-action",
                "spotting": ",".join(map(str, boxes["checkout"])), "needs_reasoning": True}),
        Sample("ui_search_box", p,
               f"Is there a search field? If so give its bounding box [x1,y1,x2,y2]. Image {W}x{H}.",
               [bx(boxes["search"])], "webui-locate", "grounding",
               {"content_class": "webui", "task": "locate", "spotting": ",".join(map(str, boxes["search"]))}),
    ]
    save_jsonl(samples, OUT / "webui.jsonl")
    im.save(OUT / "sample.png")
    (OUT / "sample.json").write_text(json.dumps({
        "benchmark": "webui_probe", "name": "Web UI/UX (agent) probe",
        "category": "F1. Custom capability axes", "metric": "grounding / exact / ned",
        "purpose": "Web-agent UI understanding: locate interactive elements (button/search/cart), "
                   "identify the primary CTA, read the nav, and reason about affordances "
                   "(what to click to act). Element boxes are exact for spotting IoU.",
        "source": "SYNTHETIC (scripts/make_webui_probe.py)",
        "ground_truth": {"n_samples": len(samples), "elements": list(boxes)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(samples)} webui samples -> {OUT/'webui.jsonl'}")


if __name__ == "__main__":
    main()
