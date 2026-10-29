"""Демонстрація ЛР1: генератори, ліниве читання, обробка зламаного кодування."""

import inspect
import logging
import sys
import tempfile
import time
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

logging.basicConfig(
    level=logging.WARNING, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s"
)
print("isgeneratorfunction(iter_documents):", inspect.isgeneratorfunction(iter_documents))
print("isgeneratorfunction(tokenize):", inspect.isgeneratorfunction(tokenize))
t0 = time.perf_counter()
d = next(iter_documents(Path("data")))
print(
    f"next(iter_documents('data')) -> doc_id={d.doc_id} title={d.title!r} "
    f"in {(time.perf_counter() - t0) * 1000:.2f} ms"
)
print(list(tokenize("Don't stop: Straße, café vs café, п'ять, event-loop 3.13")))
with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "bad.txt").write_bytes(b"Title\n=====\n\nok text \xff\xfe broken caf\xc3\xa9\n")
    for doc in iter_documents(Path(tmp)):
        print("bad file read:", doc.title, list(tokenize(doc.text)))
