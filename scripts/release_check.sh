#!/usr/bin/env bash
# Static release gate. Run from this repository's root after cloning.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail() { printf 'release_check: %s\n' "$*" >&2; exit 1; }
command -v sha256sum >/dev/null || fail "sha256sum is required"
command -v git >/dev/null || fail "git is required"

sha256sum -c SHA256SUMS
(cd patches && sha256sum -c SHA256SUMS)

# Every patch must parse as a unified diff before a user spends hours building.
# `git apply --check` needs the pinned upstream worktree, so `--stat` is the
# portable syntax gate available in this checkout.
while IFS= read -r -d '' patch; do
  git apply --stat -- "$patch" >/dev/null 2>&1 || \
    fail "unreadable patch: ${patch#$root/}"
done < <(find patches/sparkinfer -type f \( -name '*.patch' -o -name '*.diff' \) -print0)

# Reject material that must never reach a public repository. Search tracked text only,
# so this neither scans untracked models nor exposes values. Exclude this checker because
# its regex necessarily contains the literal credential prefixes it detects.
if git grep -I -n -E '(BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,})' -- . ':!scripts/release_check.sh'; then
  fail "credential-like material found in tracked text"
fi

if git ls-files | grep -Eqi '(^|/)(\.env|\.secrets/|.*\.(pem|key|p12|pfx|gguf))$'; then
  fail "private/configuration or model artifact is tracked"
fi

# These are intentionally retained as reproducibility evidence, but are not portable
# launchers. Warn loudly if a reader is about to treat them as such.
legacy_paths="$(git grep -I -l -E '(/home/victor|10\.10\.10\.|global\.prd\.ga\.run\.brev)' -- \
  scripts/spec_draft.sh scripts/spec_rollback.sh 2>/dev/null || true)"
if [[ -n "$legacy_paths" ]]; then
  printf '%s\n%s\n' \
    'release_check: NOTE historical scripts contain private lab paths; do not use them as public launchers:' \
    "$legacy_paths" >&2
fi

printf '%s\n' 'release_check: PASS (checksums, patch syntax, and secret scan)'
