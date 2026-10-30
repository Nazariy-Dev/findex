"""Локальний HTTP-сервер дзеркала (для ЛР6): статика з затримкою, як у реального сайту.

- кожна відповідь чекає LATENCY секунд (емуляція мережі: 50 мс за замовчуванням);
- кожен FAIL_EVERY-й запит отримує 429 з Retry-After: 1 (перевірка повторів);
- один URL /slow/... відповідає довше за таймаут клієнта (перевірка asyncio.timeout).
Запуск: python scripts/mirror_server.py <mirror_dir> <port> [latency] [fail_every]
"""

from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
port = int(sys.argv[2])
LATENCY = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
FAIL_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 0
counter = itertools.count(1)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            path = line.decode().split(" ")[1].split("?")[0]
            while await reader.readline() not in (b"\r\n", b""):
                pass
            n = next(counter)
            await asyncio.sleep(LATENCY)
            status, body, extra = 200, b"", ""
            if path.startswith("/slow/"):
                await asyncio.sleep(30)
            if FAIL_EVERY and n % FAIL_EVERY == 0 and path != "/robots.txt":
                status, extra, body = 429, "Retry-After: 1\r\n", b"too many requests"
            else:
                f = (root / path.lstrip("/")).resolve()
                if f.is_dir():
                    f = f / "index.html"
                if root in f.parents and f.is_file():
                    body = f.read_bytes()
                else:
                    status, body = 404, b"not found"
            ctype = "text/plain" if path.endswith(".txt") else "text/html; charset=utf-8"
            reason = {200: "OK", 404: "Not Found", 429: "Too Many Requests"}[status]
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(body)}\r\n{extra}\r\n".encode()
                + body
            )
            await writer.drain()
    except (ConnectionError, IndexError):
        pass
    finally:
        writer.close()


async def main() -> None:
    srv = await asyncio.start_server(handle, "127.0.0.1", port)
    print(
        f"serving {root} on http://127.0.0.1:{port} latency={LATENCY}s fail_every={FAIL_EVERY}",
        flush=True,
    )
    async with srv:
        await srv.serve_forever()


asyncio.run(main())
