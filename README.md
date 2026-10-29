# findex

Повнотекстовий пошуковий рушій по документації CPython (`Doc/`, 563 файли .rst) і Django (`docs/`, 680 файлів .txt).
Курс «Програмування на Python», 8 лабораторних, один репозиторій.

## Корпус

- `data/pydocs/` — `git clone --depth 1 --filter=blob:none --sparse https://github.com/python/cpython.git` + `git sparse-checkout set Doc` (коміт ef1fce0, 29.10.2026);
- `data/django/` — так само з `https://github.com/django/django.git`, `docs` (коміт 817df1c).

Лишаються лише .rst/.txt: 1243 документи, 23 МБ тексту, 2 670 986 токенів. `data/` у `.gitignore`.

## Лаба 1 — лінивий корпус

`python -m findex.stats data/ [--limit N] [--eager]`

Правила токенізатора: NFC → `casefold()` → `\w+(?:'\w+)*`; апострофи ’ ʼ зводяться до `'` і лишаються всередині слова; дефіс і крапка розривають токен; підкреслення лишається (ідентифікатори).

| Версія | Документи | Пікова пам'ять | Час |
|---|---|---|---|
| eager (lists) | 1243 | 163.2 MiB | 12.83 s |
| lazy (generators) | 1243 | 7.9 MiB | 12.31 s |

Жадібна версія тримає одночасно всі 1243 тексти (str) і 2,67 млн рядків-токенів у списках списків (кожен елемент — вказівник 8 байт плюс об'єкт str). Лінива версія тримає один документ, його токени по одному і `Counter` на 54 791 терм — це і є 7,9 MiB.

## Лаба 2 — інвертований індекс

```
python -m findex.index data/ --out index.bin [--positions] [--format pickle|binary]
python -m findex.search index.bin "asyncio event loop" [--engine merge|set]
```

`pickle.load` виконує код із файлу (через `__reduce__`), тому завантажується лише власний індекс; другий формат (binary: JSON-заголовок + буфери `array('I')`) коду не виконує.

Merge vs set, мкс на виклик (найкраще з 10 × 200):

| Пара термів | df | Операція | merge | set |
|---|---|---|---|---|
| the / in (найчастіші) | 1173 / 1197 | AND | 157.47 | 48.85 |
| the / in | 1173 / 1197 | OR | 161.03 | 49.96 |
| editability / zoomtext (найрідкісніші) | 1 / 1 | AND | 0.28 | 0.59 |
| editability / zoomtext | 1 / 1 | OR | 0.46 | 0.58 |

Збереження і завантаження (медіана з 3):

| Формат | Розмір, MiB | save, s | load, s |
|---|---|---|---|
| pickle | 9.13 | 2.79 | 3.11 |
| binary | 5.38 | 0.18 | 1.74 |
| pickle + positions | 17.18 | 3.42 | 4.02 |
| binary + positions | 17.63 | 0.74 | 2.15 |

Пам'ять трьох представлень постінгів (541 763 постінги):

| Представлення | Пік (build), MiB | Файл (pickle), MiB | load, s |
|---|---|---|---|
| `list[Posting]` plain `@dataclass` | 60.6 | 10.58 | 0.70 |
| `list[Posting]` `slots=True` | 40.1 | 8.51 | 1.58 |
| пари `array('I')` | 23.7 | 6.60 | 0.09 |

Звичайний екземпляр — 48 байт заголовка плюс окремий `__dict__` на 88 байт; зі slots словника немає. У масиві `array('I')` один постінг — 2 × 4 байти без об'єктів і без вказівників у списку. Повільніше завантаження slots-версії через pickle пояснюється тим, що dataclass зі slots відновлює стан через згенерований на Python `__setstate__`.

## Лаба 3 — рейтинг і модель об'єктів

`Index(Mapping)` підтримує `len`, `in`, `index[term]`, ітерацію і `repr`; `open_index(path)` — контекстний менеджер. TF-IDF і BM25(k1=1.5, b=0.75) задовольняють протокол `Scorer`. Парсер рекурсивного спуску будує дерево `Term/Phrase/And/Or/Not`.

Перевірки рейтингу на корпусі (`scripts/lab3_eval.py`):
1. idf(zoneinfo, df=12) = 4.600 проти idf(the, df=1173) = 0.058: документ про zoneinfo на першому місці.
2. Документ середньої довжини, терм asyncio: друге входження додає 1.190 бала, двадцяте — 0.0236.
3. tf = 1: документ на 85 токенів отримує 2.630, документ на 14 107 токенів — 0.426.

Сліди `@timed` і кешу:
```
findex.ranking: cache miss: evaluating 'python AND (async OR await) NOT java "event loop"'
findex.timing: search took 22.3 ms
findex.ranking: cache hit for 'python AND (async OR await) NOT java "event loop"'
findex.timing: search took 14.1 ms
```

Precision@5 (10 запитів, терми через OR): TF-IDF — 0.50, BM25 — 0.44.

## Лаба 4 — типи, тести, пакування, CLI

![CI](https://github.com/<user>/findex/actions/workflows/ci.yml/badge.svg)

```
uv tool install .            # або: uv tool install dist/findex-0.4.0-py3-none-any.whl
findex index data/ -o index.bin
findex search index.bin '"event loop" (thread OR executor)' -k 3
findex search index.bin 'pickle untrusted' --json | jq .path
findex stats index.bin
```

pyright strict: 0 errors; ruff: clean; pytest: 48 тестів (4 властивості Hypothesis), покриття 80%.
Поза покриттям навмисно лишились `stats.py` і `main()` у `index.py`/`search.py` (старі точки входу ЛР1–2, замінені CLI) та гілка JSONL у `corpus.py` (корпус у форматі каталогу).
