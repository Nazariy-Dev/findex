"""ЛР8, M3: precision@5 для keyword / semantic / hybrid на 10 запитах ЛР3 і 5 перефразованих.

Як і в ЛР3, для keyword терми запиту з'єднуються через OR і ранжуються BM25.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from findex.ranking import BM25, load_index, search
from findex.semantic import HybridSearcher


def main() -> None:
    path = Path(sys.argv[1])
    ix = load_index(path)
    hs = HybridSearcher.open(path)
    assert hs is not None, "run findex embed first"
    data = json.loads(Path("eval/queries.json").read_text())
    print(f"{'group':10} {'query':62} {'kw':>4} {'sem':>4} {'hyb':>4}")
    sums: dict[str, list[float]] = {}
    for group, items in data.items():
        for it in items:
            rel = set(it["rel"])
            q = it["q"]
            kw_ids = [
                r.doc_id for r in search(ix, q.replace(" ", " OR "), BM25(), k=50, snippets=False)
            ]
            sem_ids = [d for d, _ in hs.semantic(q, 5)]
            hyb_ids = [d for d, _ in hs.hybrid(kw_ids, q, 5)]
            p = [
                sum(ix.meta(d).path in rel for d in ids[:5]) / 5
                for ids in (kw_ids, sem_ids, hyb_ids)
            ]
            for name, v in zip(("kw", "sem", "hyb"), p, strict=True):
                sums.setdefault(f"{group}:{name}", []).append(v)
            print(f"{group:10} {q:62} {p[0]:4.1f} {p[1]:4.1f} {p[2]:4.1f}")
    for group in data:
        m = [sum(sums[f"{group}:{n}"]) / len(sums[f"{group}:{n}"]) for n in ("kw", "sem", "hyb")]
        print(f"{'mean ' + group:73} {m[0]:4.2f} {m[1]:4.2f} {m[2]:4.2f}")


if __name__ == "__main__":
    main()
