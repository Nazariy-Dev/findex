"""Графік прискорення від числа воркерів (дані з results/l5/*.csv)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gil_csv, ft_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def rows(p: Path) -> list[dict[str, str]]:
    with p.open() as f:
        return list(csv.DictReader(f))


g, ft = rows(gil_csv), rows(ft_csv)
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ws = [1, 2, 4, 8]
ax.plot(ws, ws, "--", color="#888", label="ідеальне лінійне")
for data, ex, label, mk in (
    (g, "threads", "потоки, GIL", "o"),
    (g, "processes", "процеси, GIL", "s"),
    (ft, "threads", "потоки, 3.13t без GIL", "^"),
):
    pts = [(int(r["workers"]), float(r["speedup"])) for r in data if r["executor"] == ex]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=label)
ax.axvline(2, color="#cc4444", lw=0.8, ls=":")
ax.text(2.1, 6.5, "os.cpu_count() = 2", color="#cc4444", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(ws, [str(w) for w in ws])
ax.set_xlabel("воркери")
ax.set_ylabel("прискорення відносно serial")
ax.set_ylim(0, 8.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(out)
print(out)
