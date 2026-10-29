"""Потоковий токенізатор.

Правила (задокументовані й покриті тестами):
1. Текст нормалізується до NFC, щоб «é» як один кодпойнт і «e + U+0301» давали один токен.
2. Регістр знімається через str.casefold() (агресивніше за lower(): «Straße» -> «strasse»).
3. Токен — послідовність символів \\w (літери будь-якої абетки, цифри, підкреслення).
   Підкреслення лишається всередині токена, бо в документації це ідентифікатори
   (get_event_loop, __init__).
4. Апостроф (' або ’ або ʼ) всередині слова зберігається і зводиться до "'":
   don't, п'ять, сім’я -> сім'я. Апостроф на краю слова відкидається.
5. Дефіс розриває слово: «event-loop» -> «event», «loop» (так запит «loop» знаходить обидва).
6. Числа — окремі токени; крапка їх розділяє: «3.13» -> «3», «13».
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})
TOKEN_RE = re.compile(r"\w+(?:'\w+)*")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().translate(APOSTROPHES)


def tokenize(text: str) -> Iterator[str]:
    """Генератор токенів: re.finditer не будує список усіх збігів."""
    for m in TOKEN_RE.finditer(normalize(text)):
        yield m.group()
