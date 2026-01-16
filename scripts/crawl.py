from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def _is_same_host(seed: str, url: str) -> bool:
    return urlparse(seed).netloc == urlparse(url).netloc


def _is_downloadable(url: str) -> bool:
    # 이미지/PDF 위주 기본형(필요 시 확장)
    return bool(re.search(r"\.(png|jpg|jpeg|webp|tif|tiff|pdf)(\?.*)?$", url, re.IGNORECASE))


def _safe_filename(url: str) -> str:
    # URL path 기반 파일명 생성
    p = urlparse(url).path
    name = Path(p).name or "download"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name


def crawl(
    *,
    seed_url: str,
    out_dir: str,
    max_pages: int,
    max_downloads: int,
    same_host_only: bool,
    sleep_sec: float,
    timeout_sec: float,
    user_agent: str,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloads_dir = out / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    visited: Set[str] = set()
    q: Deque[str] = deque([seed_url])
    downloads = 0
    pages = 0

    with meta_path.open("a", encoding="utf-8") as mf:
        while q and pages < max_pages and downloads < max_downloads:
            url = q.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                r = session.get(url, timeout=timeout_sec)
                r.raise_for_status()
            except Exception as e:
                mf.write(json.dumps({"type": "page_error", "url": url, "error": str(e)}, ensure_ascii=False) + "\n")
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            pages += 1

            if _is_downloadable(url) or "application/pdf" in ctype or ctype.startswith("image/"):
                # 직접 다운로드 타입
                fname = _safe_filename(url)
                target = downloads_dir / fname
                try:
                    target.write_bytes(r.content)
                    downloads += 1
                    mf.write(
                        json.dumps(
                            {"type": "download", "url": url, "path": str(target), "content_type": ctype},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                except Exception as e:
                    mf.write(json.dumps({"type": "download_error", "url": url, "error": str(e)}, ensure_ascii=False) + "\n")
                time.sleep(sleep_sec)
                continue

            # HTML 파싱
            if "text/html" in ctype or ctype == "" or "<html" in (r.text[:200].lower()):
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    nxt = urljoin(url, href)
                    if same_host_only and not _is_same_host(seed_url, nxt):
                        continue
                    if nxt not in visited:
                        q.append(nxt)
                # img src도 다운로드 대상으로 큐에 추가
                for img in soup.find_all("img", src=True):
                    src = urljoin(url, img["src"])
                    if same_host_only and not _is_same_host(seed_url, src):
                        continue
                    if src not in visited:
                        q.append(src)

            time.sleep(sleep_sec)


def main() -> None:
    p = argparse.ArgumentParser(description="Simple crawler for downloading PDFs/images + metadata 기록")
    p.add_argument("--seed_url", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_pages", type=int, default=50)
    p.add_argument("--max_downloads", type=int, default=200)
    p.add_argument("--same_host_only", type=int, default=1)
    p.add_argument("--sleep_sec", type=float, default=0.5)
    p.add_argument("--timeout_sec", type=float, default=15.0)
    p.add_argument("--user_agent", type=str, default="OCR-Crawler/0.1")
    args = p.parse_args()

    crawl(
        seed_url=args.seed_url,
        out_dir=args.out_dir,
        max_pages=args.max_pages,
        max_downloads=args.max_downloads,
        same_host_only=bool(args.same_host_only),
        sleep_sec=args.sleep_sec,
        timeout_sec=args.timeout_sec,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()


