#!/usr/bin/env bash
# Rsync the local news DB to prod and restart the service. Run from cron.
# Usage: scripts/sync-db.sh [path-to-db]   (default: agent/news_data.db)
# Host, user, and key come from the repo-root .envrc (cron doesn't load direnv).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
[ -f "$REPO_ROOT/.envrc" ] && source "$REPO_ROOT/.envrc"
set -u

TARGET_HOST="${HOST_IP:-${DEPLOY_HOST:?set HOST_IP or DEPLOY_HOST in .envrc}}"
: "${SSH_KEY:?set SSH_KEY in .envrc}"
DEPLOY_USER="${DEPLOY_USER:-ec2-user}"
DB_SOURCE="${1:-$REPO_ROOT/agent/news_data.db}"
LOG_FILE="$REPO_ROOT/upload.log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

rsync -avz --progress -e "ssh -i $SSH_KEY" "$DB_SOURCE" \
  "$DEPLOY_USER@$TARGET_HOST:/opt/news/data/news.db" >> "$LOG_FILE" 2>&1

ssh -i "$SSH_KEY" "$DEPLOY_USER@$TARGET_HOST" 'sudo systemctl restart news' >> "$LOG_FILE" 2>&1

echo "Done." >> "$LOG_FILE"
