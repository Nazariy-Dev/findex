"""ЛР5: окремо час map (воркери) і merge (послідовний) — частка Амдала."""

from __future__ import annotations

import time
from pathlib import Path

from findex.parallel import build_parallel


def main() -> None:
    for ex, w in (("serial", 1), ("threads", 2), ("processes", 2)):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            _, tm = build_parallel(Path("data"), w, ex, positions=True)  # type: ignore[arg-type]
            ts.append((tm.map_s, tm.merge_s, time.perf_counter() - t))
        ts.sort(key=lambda x: x[2])
        m, mg, tot = ts[1]  # медіана за загальним часом
        print(
            f"{ex:9} workers={w}  map={m:.2f}s  merge={mg:.2f}s  total={tot:.2f}s  "
            f"merge share={mg / tot:.0%}"
        )


if __name__ == "__main__":  # обов'язково під spawn: дочірній процес імпортує цей модуль
    main()
