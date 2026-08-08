#!/usr/bin/env bash
#
# Verify that every place declaring the project's version agrees.
#
# The Rust workspace version is the single source of truth. Everything else —
# the internal dependency pins, the npm package, and the pinned examples in the
# README — is checked against it. On a release branch or tag the name itself is
# checked too, so `release/v0.3.3-*` can never ship 0.3.2 artifacts.
#
# Usage:
#   scripts/check-release-version.sh              # infer from branch/tag
#   scripts/check-release-version.sh 0.3.3        # check against an explicit version
#
# Exits non-zero listing every mismatch.

set -uo pipefail

cd "$(dirname "$0")/.."

CARGO_TOML="Cargo.toml"
NPM_PKG="packages/lattice-db-js/package.json"
README="README.md"

failures=0
checks=0

pass() { checks=$((checks + 1)); printf '  \033[32m✓\033[0m %-46s %s\n' "$1" "$2"; }
fail() {
  checks=$((checks + 1))
  failures=$((failures + 1))
  printf '  \033[31m✗\033[0m %-46s %s\n' "$1" "$2"
}

# --- Reference version -------------------------------------------------------

# The workspace version: the first `version = "..."` inside [workspace.package].
workspace_version=$(
  awk '
    /^\[workspace\.package\]/ { in_section = 1; next }
    /^\[/                     { in_section = 0 }
    in_section && /^version[[:space:]]*=/ {
      match($0, /"[^"]+"/); print substr($0, RSTART + 1, RLENGTH - 2); exit
    }
  ' "$CARGO_TOML"
)

if [ -z "$workspace_version" ]; then
  echo "error: could not read [workspace.package] version from $CARGO_TOML" >&2
  exit 2
fi

# --- Expected version: explicit arg, else the tag/branch name ----------------

# Extracts 1.2.3 from: v1.2.3, release/v1.2.3, release/v1.2.3-anything
version_from_ref() {
  printf '%s' "$1" | sed -nE 's#^(release/)?v([0-9]+\.[0-9]+\.[0-9]+)(-.*)?$#\2#p'
}

expected="${1:-}"
source_desc="explicit argument"

if [ -z "$expected" ]; then
  # On a pull request GITHUB_REF_NAME is the merge ref ("18/merge"), so the
  # head branch has to come from GITHUB_HEAD_REF for release branches to be
  # recognised. Falls back to the checked-out branch outside CI.
  ref="${GITHUB_HEAD_REF:-}"
  [ -z "$ref" ] && ref="${GITHUB_REF_NAME:-}"
  [ -z "$ref" ] && ref="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  from_ref=$(version_from_ref "$ref")
  if [ -n "$from_ref" ]; then
    expected="$from_ref"
    source_desc="ref '$ref'"
  else
    # Not a release ref: still verify everything agrees with the workspace.
    expected="$workspace_version"
    source_desc="workspace version (ref '$ref' is not a release ref)"
  fi
fi

echo "Expected version: $expected  (from $source_desc)"
echo

# --- Authoritative declarations ---------------------------------------------

echo "Manifests"

if [ "$workspace_version" = "$expected" ]; then
  pass "Cargo.toml [workspace.package] version" "$workspace_version"
else
  fail "Cargo.toml [workspace.package] version" "$workspace_version (expected $expected)"
fi

# Internal crates are pinned by version in [workspace.dependencies] so that
# `cargo publish` can resolve them; those pins must track the workspace version.
# Only deps that actually carry a version are checked: lattice-test-harness is
# deliberately version-less so `cargo publish` strips it (it is unpublished).
while IFS= read -r line; do
  [ -z "$line" ] && continue
  crate=$(printf '%s' "$line" | sed -nE 's/.*package = "([^"]+)".*/\1/p')
  pinned=$(printf '%s' "$line" | sed -nE 's/.*version = "([^"]+)".*/\1/p')
  label="Cargo.toml dep pin ($crate)"
  if [ "$pinned" = "$expected" ]; then
    pass "$label" "$pinned"
  else
    fail "$label" "${pinned:-<none>} (expected $expected)"
  fi
done <<EOF
$(grep -E '^[a-z-]+ = \{ path = "crates/' "$CARGO_TOML" | grep 'version = "')
EOF

npm_version=$(sed -nE 's/^[[:space:]]*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$NPM_PKG" | head -1)
if [ "$npm_version" = "$expected" ]; then
  pass "packages/lattice-db-js/package.json" "$npm_version"
else
  fail "packages/lattice-db-js/package.json" "${npm_version:-<none>} (expected $expected)"
fi

# --- Documentation pins ------------------------------------------------------
#
# Stale install instructions are a real defect: they send users to assets that
# do not exist for the current release.

echo
echo "Documented pins ($README)"

# Reports every match of `pattern` that does not contain `expected_text`.
# The filter is a fixed string (-F) and guarded with `--`, since the expected
# text can begin with a dash (e.g. "-v0.3.3").
check_readme_pattern() {
  local label="$1" pattern="$2" expected_text="$3"
  local found
  found=$(grep -noE "$pattern" "$README" | grep -vF -- "$expected_text" || true)
  if [ -z "$found" ]; then
    pass "$label" "$expected_text"
  else
    fail "$label" "stale: $(printf '%s' "$found" | tr '\n' ' ')"
  fi
}

# Release-asset download URLs, e.g. .../releases/download/v0.3.3/...
check_readme_pattern "release download URLs" \
  "download/v[0-9]+\.[0-9]+\.[0-9]+" "download/v$expected"

# Pinned asset filenames, e.g. lattice-server-linux-x64-v0.3.3.
# The character class allows digits so platform segments like "x64" match.
check_readme_pattern "pinned asset filenames" \
  "lattice[a-z0-9-]*-v[0-9]+\.[0-9]+\.[0-9]+" "-v$expected"

# npm/CDN pins. Matches both the qualified form (lattice-db@0.3.3) and the
# bare form used in prose ("swap @latest for `@0.3.3`").
check_readme_pattern "npm / CDN version pins" \
  "@[0-9]+\.[0-9]+\.[0-9]+" "@$expected"

# git dependency tag, e.g. tag = "v0.3.3"
check_readme_pattern "git dependency tag" \
  "tag = \"v[0-9]+\.[0-9]+\.[0-9]+\"" "tag = \"v$expected\""

# The crates.io example uses a MAJOR.MINOR range, so only those must match.
expected_range="${expected%.*}"
range_found=$(grep -noE 'latticedb-[a-z]+ = \{ version = "[0-9]+\.[0-9]+"' "$README" \
  | grep -v "\"$expected_range\"" || true)
if [ -z "$range_found" ]; then
  pass "crates.io dependency range" "$expected_range"
else
  fail "crates.io dependency range" "stale: $(printf '%s' "$range_found" | tr '\n' ' ')"
fi

# --- Result ------------------------------------------------------------------

echo
if [ "$failures" -eq 0 ]; then
  echo "All $checks version declarations agree on $expected."
  exit 0
fi

echo "$failures of $checks version declarations disagree with $expected." >&2
echo "Update them (or pass the intended version explicitly) before releasing." >&2
exit 1
