#!/usr/bin/env bash
# One-time setup for a personal clone: hooks, rebase-by-default, identity check.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
git config pull.rebase true
git config rebase.autoStash true
git config rebase.autoSquash true
git config push.autoSetupRemote true

if [[ -z $(git config user.name) || -z $(git config user.email) ]]; then
  echo "Set your identity first: git config --global user.name/user.email" >&2
  exit 1
fi
echo "dev setup done for $(git config user.name) in $PWD"
