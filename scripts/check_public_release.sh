#!/usr/bin/env bash
set -euo pipefail

failed=0

artifact_re='(^|/)(answers|answers_upload|results|outputs?|logs|eval_submissions|aihc_scripts|rebuttal)(/|$)|\.xlsx$|\.jsonl$|wrong_answers\.tsv$|bot[0-9]*_monitor_status\.md$'
artifact_files=$(git ls-files | grep -E "$artifact_re" || true)
if [[ -n "$artifact_files" ]]; then
  printf 'ERROR: generated or internal artifacts are tracked:\n%s\n' "$artifact_files" >&2
  failed=1
fi

# Split sensitive prefixes in this source so the checker does not match itself.
secret_re='AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|s''k-[A-Za-z0-9_-]{20,}|h''f_[A-Za-z0-9]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
secret_files=$(git grep -Il -E "$secret_re" -- . ':!scripts/check_public_release.sh' || true)
if [[ -n "$secret_files" ]]; then
  printf 'ERROR: credential-like material found in tracked files:\n%s\n' "$secret_files" >&2
  failed=1
fi

internal_re='/mnt/eason|/mnt/gyc|cce-pmm|10\.40\.|192\.168\.|\.aihc|AIHC_'
internal_files=$(git grep -Il -E "$internal_re" -- . ':!scripts/check_public_release.sh' || true)
if [[ -n "$internal_files" ]]; then
  printf 'ERROR: private infrastructure references found in tracked files:\n%s\n' "$internal_files" >&2
  failed=1
fi

if (( failed )); then
  exit 1
fi

printf 'PUBLIC_RELEASE_CHECK_OK tracked_files=%s\n' "$(git ls-files | wc -l | tr -d ' ')"
