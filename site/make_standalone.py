"""
make_standalone.py — 產生單一 HTML 檔版本
=========================================
把 assets/ 裡的 CSS 和圖片全部轉成 base64 內嵌到 HTML 中。

為什麼需要這個？
---------------
Neocities 免費方案不允許上傳圖片檔，API 會回覆：
    "file is not an allowed file type for free sites, supporter required"

所以不能把 .jpg 當成獨立檔案上傳。改成把圖片編碼成 data URI
直接寫進 HTML，整份報告就只需要上傳一個檔案。

執行
----
    python make_standalone.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
ASSETS = SITE / "assets"


def to_data_uri(path: Path) -> str:
    """圖片檔轉成 base64 data URI。"""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    html_path = SITE / "index.html"
    css_path = ASSETS / "report.css"

    if not html_path.exists():
        print(f"[ERROR] 找不到 {html_path}")
        return 1

    html = html_path.read_text(encoding="utf-8")

    # ── 1. 內嵌 CSS ──
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        html = html.replace(
            '<link rel="stylesheet" href="assets/report.css">',
            f"<style>\n{css}\n</style>",
        )
        print(f"[OK] CSS 已內嵌（{len(css):,} 字元）")

    # ── 2. 內嵌圖片 ──
    embedded = 0
    skipped = 0
    for img in sorted(ASSETS.glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            continue

        rel = f"assets/{img.name}"
        if rel not in html:
            skipped += 1
            continue

        uri = to_data_uri(img)
        html = html.replace(f'src="{rel}"', f'src="{uri}"')
        embedded += 1
        print(f"[OK] {img.name:<26s} → base64 "
              f"({img.stat().st_size / 1024:>6.1f} KB)")

    # ── 3. 清掉殘留的外部 CSS 引用 ──
    html = re.sub(r'<link rel="stylesheet" href="assets/[^"]*">', "", html)

    # ── 4. 移除佔位框（單檔版不需要圖片檔，佔位會顯得奇怪）──
    # 保留佔位框反而有用（提示使用者哪裡還缺圖），所以不清

    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")

    size_mb = out.stat().st_size / 1048576
    print()
    print("=" * 60)
    print(f"完成：{out.name}  {size_mb:.2f} MB  （內嵌 {embedded} 張圖）")
    if skipped:
        print(f"      另有 {skipped} 個圖片檔未被引用（可刪除）")
    print("=" * 60)
    print()
    print("  這個 index.html 已經是完全獨立的檔案，")
    print("  直接上傳到 Neocities 即可，不需要其他檔案。")
    return 0


if __name__ == "__main__":
    sys.exit(main())