#!/usr/bin/env bash
# Збірка на Render (runtime python): залежності, корпус, індекс, ембединги.
# Корпус не зберігається в репозиторії (data/ у .gitignore), тому завантажується під час build
# з тих самих джерел, що й локально: розрідженим git clone документації CPython і Django.
set -euo pipefail
pip install --quiet uv
uv sync --locked --no-dev --extra semantic
mkdir -p deploy/data
clone() {  # $1 = url, $2 = підкаталог у репозиторії, $3 = ціль у deploy/data
  rm -rf /tmp/src && git clone --quiet --depth 1 --filter=blob:none --sparse "$1" /tmp/src
  git -C /tmp/src sparse-checkout set "$2"
  rm -rf "deploy/data/$3" && cp -r "/tmp/src/$2" "deploy/data/$3"
}
clone https://github.com/python/cpython.git Doc pydocs
clone https://github.com/django/django.git docs django
find deploy/data -type f ! \( -name '*.rst' -o -name '*.txt' \) -delete
uv run --no-sync findex index deploy/data -o deploy/index.bin
uv run --no-sync findex embed deploy/index.bin
