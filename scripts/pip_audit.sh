#!/usr/bin/env bash
# Audits the installed environment against known advisories (used by `make audit` and CI).
# Advisories listed in .pip-audit-ignore are accepted, with their reason recorded there;
# anything else exits non-zero. PIP_AUDIT_IGNORE_FILE overrides the list (e.g. /dev/null).
set -euo pipefail

cd "$(dirname "$0")/.."
ignore_file="${PIP_AUDIT_IGNORE_FILE:-.pip-audit-ignore}"

args=()
while read -r id _; do
  case "$id" in
    "" | \#*) continue ;;
  esac
  args+=(--ignore-vuln "$id")
done <"$ignore_file"

# ${args[@]+...}: an empty array is an unbound variable under `set -u` in macOS's bash 3.2.
exec pip-audit --skip-editable --progress-spinner off ${args[@]+"${args[@]}"}
