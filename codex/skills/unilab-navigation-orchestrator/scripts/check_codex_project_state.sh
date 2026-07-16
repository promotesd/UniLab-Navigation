#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/xiaodudu/robot_study/UniLab-Navigation}"
cd "${REPO_ROOT}"

echo "Repository: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo
echo "Status:"
git status --short
echo
echo "Recent commits:"
git log -8 --oneline
echo
echo "Navigation files:"
find src/unilab/envs/navigation tests/envs/navigation \
  -maxdepth 4 -type f 2>/dev/null | sort
