#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: install_overlay.sh /path/to/clean/llava}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

test -d "$TARGET/llava/model"
test -d "$TARGET/llava/eval"
cp -R "$HERE/llava/." "$TARGET/llava/"
printf 'Installed LLaVA STAR-Pro overlay into %s\n' "$TARGET"
