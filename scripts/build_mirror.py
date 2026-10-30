"""Локальне HTML-дзеркало корпусу для ЛР6: кожен .rst/.txt -> сторінка з посиланнями.

Директиви Sphinx (:func:, .. method::) docutils не знає, тому сторінки рендеряться
як <pre> з текстом документа, а посилання ставляться на 8 сусідніх документів того
самого каталогу і на індекс каталогу. Так виходить зв'язний граф на 1243 сторінки.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from findex.corpus import guess_title, iter_paths, read_text_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
paths = [p.relative_to(src) for p in iter_paths(src)]
by_dir: dict[Path, list[Path]] = {}
for p in paths:
    by_dir.setdefault(p.parent, []).append(p)


def page_url(p: Path) -> str:
    return "/" + p.with_suffix(".html").as_posix()


for p in paths:
    text = read_text_file(src / p)
    title = guess_title(text, p.stem)
    sib = by_dir[p.parent]
    j = sib.index(p)
    neigh = sib[max(0, j - 4) : j] + sib[j + 1 : j + 5]
    links = "".join(f'<li><a href="{page_url(n)}">{html.escape(n.stem)}</a></li>' for n in neigh)
    body = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"</head><body><nav><a href='/index.html'>home</a> <a href='{page_url(p.parent / 'index')}"
        f"#top'>section</a></nav><h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        f"<ul>{links}</ul><footer>mirror of {p.as_posix()}</footer></body></html>"
    )
    out = dst / p.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")

for d, items in by_dir.items():
    sub = sorted(
        {
            q.parts[len(d.parts)]
            for q in paths
            if q.parts[: len(d.parts)] == d.parts and len(q.parts) > len(d.parts) + 1
        }
    )
    lis = "".join(f'<li><a href="{page_url(q)}">{html.escape(q.stem)}</a></li>' for q in items)
    lis += "".join(f'<li><a href="/{(d / s).as_posix()}/index.html">{s}/</a></li>' for s in sub)
    idx = dst / d / "index.html"
    if (src / d / "index.rst").exists() or (src / d / "index.txt").exists():
        idx = dst / d / "toc.html"  # не затираємо сторінку самого index.rst
        lis = f'<li><a href="{page_url(d / "index")}">index</a></li>' + lis
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(
        f"<!doctype html><html><head><title>{d.as_posix() or 'home'}</title></head>"
        f"<body><ul>{lis}</ul></body></html>",
        encoding="utf-8",
    )
root_links = "".join(
    f'<li><a href="/{s}/toc.html">{s}</a></li>'
    if (dst / s / "toc.html").exists()
    else f'<li><a href="/{s}/index.html">{s}</a></li>'
    for s in sorted({p.parts[0] for p in paths if len(p.parts) > 1})
)
(dst / "index.html").write_text(
    f"<html><head><title>mirror</title></head><body><ul>{root_links}</ul></body></html>",
    encoding="utf-8",
)
(dst / "robots.txt").write_text("User-agent: *\nDisallow: /pydocs/whatsnew/\n", encoding="utf-8")
print(f"{len(paths)} pages -> {dst}")
