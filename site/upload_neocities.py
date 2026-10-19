"""
upload_neocities.py — 上傳檔案到 Neocities
=========================================
用 Neocities 官方 API 上傳。

API 重點（2026 新版）
--------------------
1. 認證走 HTTP 標頭，不是表單欄位：
       Authorization: Bearer <API_KEY>
2. /api/upload 用 multipart/form-data，而且「欄位名稱就是目標路徑」：
       -F "20261014/index.html=@index.html"
   不是舊版的 `-F file=@x` 加一個 `path` 參數。

用法
----
    python upload_neocities.py                         # index.html → 20261014/index.html
    python upload_neocities.py --to index.html          # 改傳到首頁
    python upload_neocities.py --to css/style.css --file assets/report.css

注意
----
免費方案不允許上傳圖片檔（JPG/PNG/GIF），會回覆
    "file is not an allowed file type for free sites, supporter required"
所以圖片一律用 base64 內嵌在 HTML 裡，只上傳 .html 單一檔案。

API 金鑰
--------
在 https://neocities.org/settings/<站台名稱>#api_key 產生。
等同密碼，勿外流。
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 需要 requests：python -m pip install requests")
    sys.exit(1)

API_UPLOAD = "https://neocities.org/api/upload"
API_LIST = "https://neocities.org/api/list"
API_KEY = "b05002b109277000f41be577246b7906"   # voidfalse 站台

SITE = Path(__file__).resolve().parent


def upload(local: Path, to: str, api_key: str = API_KEY) -> tuple[bool, str]:
    """
    上傳單一檔案。

    參數
    ----
    local : 本機檔案
    to    : 站台上的目標路徑，例如 20261014/index.html（不用開頭斜線）

    回傳
    ----
    (成功?, 訊息)
    """
    if not local.exists():
        return False, f"本機檔案不存在：{local}"

    # 目標路徑正規化：去掉開頭斜線
    to = to.lstrip("/")
    mime = mimetypes.guess_type(local.name)[0] or "application/octet-stream"

    try:
        with local.open("rb") as fh:
            # 重點：欄位名稱 = 目標路徑
            resp = requests.post(
                API_UPLOAD,
                headers={"Authorization": f"Bearer {api_key}"},
                files={to: (Path(to).name, fh, mime)},
                timeout=300,
            )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    return (resp.ok, f"HTTP {resp.status_code}\n{resp.text[:800]}")


def list_files(path: str = "", api_key: str = API_KEY) -> str:
    """列出站台上的檔案，用來確認上傳結果。"""
    params = {"path": path} if path else {}
    try:
        r = requests.get(API_LIST, headers={"Authorization": f"Bearer {api_key}"},
                         params=params, timeout=60)
        if not r.ok:
            return f"  HTTP {r.status_code}: {r.text[:200]}"

        lines = []
        for f in r.json().get("files", []):
            if f["is_directory"]:
                lines.append(f"  [dir] {f['path']}")
            else:
                lines.append(f"  [{f.get('size', 0):>8,} B]  {f['path']}")
        return "\n".join(lines) or "  (空)"
    except Exception as e:
        return f"  {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="上傳檔案到 Neocities")
    ap.add_argument("--file", default=str(SITE / "index.html"),
                    help="本機檔案（預設 site/index.html）")
    ap.add_argument("--to", default="20261014/index.html",
                    help="站台路徑（預設 20261014/index.html）")
    args = ap.parse_args()

    local = Path(args.file)
    size_kb = local.stat().st_size / 1024 if local.exists() else 0

    print(f"本機檔案：{local.name}  ({size_kb:,.1f} KB)")
    print(f"目標路徑：{args.to.lstrip('/')}")
    print("-" * 56)

    ok, msg = upload(local, args.to)
    print(msg)
    print("-" * 56)

    if ok:
        folder = "/" + args.to.lstrip("/").rsplit("/", 1)[0] if "/" in args.to else "/"
        print(f"上傳成功 → https://voidfalse.neocities.org{folder}")
        print()
        print(f"站台 {folder} 目前內容：")
        print(list_files(args.to.lstrip('/').rsplit("/", 1)[0] if "/" in args.to else ""))
    else:
        print("上傳失敗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())