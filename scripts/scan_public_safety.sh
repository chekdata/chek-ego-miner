#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"

if ! command -v rg >/dev/null 2>&1; then
  echo "scan_public_safety: ripgrep (rg) is required" >&2
  exit 1
fi

declare -a PATTERNS=(
  '100\.[0-9]+\.[0-9]+\.[0-9]+'
  'tailscale'
  'edge-local-debug-token'
  'Authorization: Bearer'
  'MacBook-Pro-9'
  'edge-ubuntu'
  'chekkk-X670'
)

has_findings=0

for pattern in "${PATTERNS[@]}"; do
  if rg -n \
    --hidden \
    --glob '!/.git' \
    --glob '!**/.git/**' \
    --glob '!scripts/scan_public_safety.sh' \
    "$pattern" "$ROOT_DIR"; then
    has_findings=1
  fi
done

if [[ "$has_findings" -ne 0 ]]; then
  echo "scan_public_safety: found potential internal-only content" >&2
  exit 1
fi

echo "scan_public_safety: no internal-host/token patterns found"
