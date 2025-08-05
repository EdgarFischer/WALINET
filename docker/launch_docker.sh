#!/usr/bin/env bash
set -euo pipefail

read -rp "Activate Git integration? (y/N) " ENABLE_GIT
WORKSPACE="$(realpath ../../)"

if [[ "$ENABLE_GIT" =~ ^[Yy] ]]; then
  echo "→ Starting SSH agent and loading key…"
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_rsa

  if [ -z "${SSH_AUTH_SOCK:-}" ]; then
    echo "✗ SSH_AUTH_SOCK is empty – connect with agent‑forwarding (ssh -A …)."
    exit 1
  fi
  echo "✓ Using SSH_AUTH_SOCK: ${SSH_AUTH_SOCK}"

  GIT_NAME="$(git config --get user.name)"
  GIT_EMAIL="$(git config --get user.email)"

  echo "→ Launching container with Git support…"
  docker run -d \
    --gpus all \
    --name mrjo \
    -p 8888:8888 \
    -v "${SSH_AUTH_SOCK}:/ssh-agent.sock" \
    -v "${HOME}/.ssh:/home/hostuser/.ssh:ro" \
    -e SSH_AUTH_SOCK=/ssh-agent.sock \
    -e GIT_AUTHOR_NAME="${GIT_NAME}" \
    -e GIT_AUTHOR_EMAIL="${GIT_EMAIL}" \
    -e GIT_COMMITTER_NAME="${GIT_NAME}" \
    -e GIT_COMMITTER_EMAIL="${GIT_EMAIL}" \
    -v "${WORKSPACE}:/workspace" \
    dl

  echo "→ Configuring Git safe.directory…"
  docker exec -u hostuser mrjo \
    git config --global --add safe.directory /workspace/Deuterium_Reconstruction

  echo "✅ Container 'mrjo' with Git is up and running."
else
  echo "→ Launching container without Git…"
  docker run -d \
    --gpus all \
    --name mrjo \
    -p 8888:8888 \
    -v "${WORKSPACE}:/workspace" \
    dl

  echo "✅ Container 'mrjo' without Git is up and running."
fi

