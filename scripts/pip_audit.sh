#!/usr/bin/env bash
# Audits the installed environment against known advisories (used by `make audit` and CI).
# Advisories listed in .pip-audit-ignore are accepted, with their reason recorded there;
# anything else exits non-zero, as does an accepted ID that is no longer reported (needs jq).
# PIP_AUDIT_IGNORE_FILE overrides the list (e.g. /dev/null).
set -euo pipefail

cd "$(dirname "$0")/.."
ignore_file="${PIP_AUDIT_IGNORE_FILE:-.pip-audit-ignore}"

args=()
ignored=""
while read -r id _; do
  case "$id" in
    "" | \#*) continue ;;
  esac
  args+=(--ignore-vuln "$id")
  ignored+="$id"$'\n'
done <"$ignore_file"

# An accepted ID that pip-audit no longer reports (by ID or alias) is stale: the package moved
# past it, and leaving it would hide a later advisory with the same ID. Fail so it is removed.
if [[ -n $ignored ]]; then
  report="$(pip-audit --skip-editable --progress-spinner off -f json)" || true
  if ! jq -e '.dependencies | type == "array"' <<<"$report" >/dev/null 2>&1; then
    echo "pip-audit did not return a JSON report; cannot check $ignore_file for stale entries" >&2
    exit 1
  fi
  reported="$(jq -r '.dependencies[].vulns[]? | .id, .aliases[]?' <<<"$report")"
  stale=0
  while read -r id; do
    [[ -n $id ]] || continue
    if ! grep -qxF "$id" <<<"$reported"; then
      echo "$id is no longer reported; remove it from $ignore_file" >&2
      stale=1
    fi
  done <<<"$ignored"
  [[ $stale == 0 ]] || exit 1
fi

# ${args[@]+...}: an empty array is an unbound variable under `set -u` in macOS's bash 3.2.
exec pip-audit --skip-editable --progress-spinner off ${args[@]+"${args[@]}"}
