#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github-username> [repo-name]"
  exit 1
fi

USER_NAME="$1"
REPO_NAME="${2:-Network_Malicious_Detection_340W}"
TARGET="${USER_NAME}/${REPO_NAME}"

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login"
  exit 1
fi

if git remote get-url upstream >/dev/null 2>&1; then
  :
elif git remote get-url origin >/dev/null 2>&1; then
  ORIGIN_URL="$(git remote get-url origin)"
  if [[ "$ORIGIN_URL" == "https://github.com/tmushd/Network_Malicious_Detection.git" ]]; then
    git remote rename origin upstream
  fi
fi

if ! git remote get-url student-origin >/dev/null 2>&1; then
  gh repo create "$TARGET" --public --source=. --remote=student-origin --push
else
  git push -u student-origin HEAD
fi

echo "Published: https://github.com/${TARGET}"

