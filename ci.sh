#!/usr/bin/env bash
# ci.sh — kit-ffi CI pipeline for the bupha shell runner.
#
# Same layout model as kernel/scripts/ci.sh: bupha invokes this via
# `runner.kind = "script"` → `bash ci.sh` inside a detached worktree of
# the pushed sha; it also works standalone from the main checkout.
#
#   ci.sh           # the whole gate: check + test
#   ci.sh doctor    # toolchain + layout smoke, no compile
#   ci.sh check     # cargo check --workspace --all-targets
#   ci.sh test      # cargo test --workspace --no-fail-fast
#
# Env:
#   CI_JOBS          cargo test threads (default 2 — the box is
#                    memory-bound; do NOT raise)
#   CI_BUILD_JOBS    cargo build -j (default 2)
#   CI_TARGET_DIR    shared cargo target dir
#                    (default <workspace>/.ci-target/kit-ffi)
#   HANDL_TOKEN / BUPHA_TOKEN  GitHub token for private git deps. Optional
#                    locally: every handl-lang dep is [patch]'d to a
#                    sibling checkout, so cargo never hits github for them.
set -euo pipefail

cd "$(dirname "$0")"

JOBS="${CI_JOBS:-2}"
BUILD_JOBS="${CI_BUILD_JOBS:-2}"

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
# Cargo must shell out to git so the insteadOf credential rewrite and
# ssh/config auth apply to private git deps.
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export CARGO_BUILD_JOBS="$BUILD_JOBS"
export RUST_TEST_THREADS="$JOBS"

# ---------------------------------------------------------------------------
# Layout plumbing — the committed [patch] tables are relative paths
# (../engine-*, ../host, ../kernel, ../kit-*), and sibling manifests reach
# this repo transitively as ../kit-ffi. In a detached worktree recreate the
# sibling layout in the worktree's parent: ../kit-ffi -> this worktree (the
# sha under test), every other sibling -> the real checkout under
# ~/GitHub/handl-lang. A stale/absent sibling breaks dep resolution before
# a single crate compiles.
# ---------------------------------------------------------------------------

# Workspace root = parent of the *main* checkout (correct from both the
# checkout and any detached worktree of this repo).
GIT_COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"
MAIN_CHECKOUT="$(dirname "$GIT_COMMON")"
WORKSPACE_ROOT="$(dirname "$MAIN_CHECKOUT")"
IN_WORKTREE=0
[ "$PWD" != "$MAIN_CHECKOUT" ] && IN_WORKTREE=1

# Shared target dir — a detached worktree's own target/ would cold-build
# every sha. Runs are serialized by the watcher (single worker,
# latest-wins), so sharing is safe. Prune with `cargo clean` under
# CI_TARGET_DIR if disk gets tight.
export CARGO_TARGET_DIR="${CI_TARGET_DIR:-$WORKSPACE_ROOT/.ci-target/kit-ffi}"

ensure_siblings() {
  [ "$IN_WORKTREE" = 1 ] || return 0
  local parent; parent="$(cd .. && pwd)"
  if [ "$parent" = "$WORKSPACE_ROOT" ]; then
    echo "ci.sh: CI worktree sits inside $WORKSPACE_ROOT — ../kit-ffi would" >&2
    echo "       resolve to the dev checkout instead of this sha. Move the" >&2
    echo "       worktree base outside the workspace root." >&2
    return 1
  fi
  # Self: ../kit-ffi must name THIS tree — transitive sibling path deps on
  # kit-ffi crates must build the pushed sha, not the dev checkout. Only
  # ever (re)point a symlink; refuse to clobber a real directory.
  if [ -e "$parent/kit-ffi" ] && [ ! -L "$parent/kit-ffi" ]; then
    echo "ci.sh: $parent/kit-ffi is a real directory — refusing to clobber" >&2
    return 1
  fi
  ln -sfn "$PWD" "$parent/kit-ffi"
  local d name
  for d in "$WORKSPACE_ROOT"/*/; do
    name="$(basename "$d")"
    [ "$name" = "kit-ffi" ] && continue
    # refresh symlinks (a previous run's links may be stale or dangling);
    # never clobber a real directory that happens to sit in the parent.
    if [ -L "$parent/$name" ] || [ ! -e "$parent/$name" ]; then
      ln -sfn "$d" "$parent/$name"
    fi
  done
}

# Private git deps: insteadOf rewrites scoped to this process tree via
# GIT_CONFIG_* env — never touches ~/.gitconfig. A token wins when the
# watcher exports one; otherwise rewrite ssh URLs to https so the gh
# credential helper / ssh config can authenticate.
setup_auth() {
  local tok="${HANDL_TOKEN:-${BUPHA_TOKEN:-}}"
  if [ -n "$tok" ]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="url.https://x-access-token:${tok}@github.com/.insteadOf"
    export GIT_CONFIG_VALUE_0="https://github.com/"
  elif [ -z "${GIT_CONFIG_COUNT:-}" ]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="url.git@github.com:handl-lang/.insteadOf"
    export GIT_CONFIG_VALUE_0="https://github.com/handl-lang/"
  fi
}

# sccache — optional shared backend; the env file is the same one the
# dagger runner mounted as a secret.
setup_sccache() {
  command -v sccache >/dev/null 2>&1 || return 0
  export RUSTC_WRAPPER=sccache
  if [ -f "$HOME/.config/bupha/sccache-env" ]; then
    set -a; . "$HOME/.config/bupha/sccache-env"; set +a
  fi
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

stage_doctor() {
  rustc --version
  cargo --version
  sccache --version 2>/dev/null || echo "sccache: not installed"
  printf 'patch tables in Cargo.toml: %s\n' "$(grep -c 'patch\.' Cargo.toml || true)"
  echo "workdir:   $PWD"
  echo "worktree:  $IN_WORKTREE (main checkout: $MAIN_CHECKOUT)"
  echo "target:    $CARGO_TARGET_DIR"
  local missing=0 d name
  for name in $(grep -o 'path = "\.\./[^"/]*' Cargo.toml | sed 's|.*\.\./||' | sort -u); do
    d="$(cd .. && pwd)/$name"
    if [ ! -d "$d" ]; then
      echo "siblings:  ../$name MISSING"
      missing=1
    fi
  done
  [ "$missing" = 0 ] && echo "siblings:  all ../<repo> patch targets present"
  return "$missing"
}

stage_check() {
  cargo check --workspace --all-targets
}

stage_test() {
  # Warm the cargo caches serially before the build fan-out.
  cargo fetch
  cargo test --workspace --no-fail-fast -- --test-threads="$JOBS"
}

stage_ci() {
  stage_check && stage_test && echo "ci passed: check + test"
}

# ---------------------------------------------------------------------------

stage="ci"
if [ $# -gt 0 ]; then stage="$1"; shift; fi

setup_auth
ensure_siblings || exit 1   # wrong layout = wrong sha under test; refuse
setup_sccache

case "$stage" in
  doctor)          stage_doctor ;;
  check)           stage_check ;;
  test)            stage_test ;;
  ci)              stage_ci ;;
  -h|--help|help)  sed -n '2,23p' "$0" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
