#!/usr/bin/env bash
set -euo pipefail

DB_SOURCE="/home/brad/projects/robot-overlord-news/agent/news_data.db"
DB_DEST="ec2-user@44.221.25.226:/opt/news/data/news.db"
SSH_KEY="$HOME/.ssh/robot-overlord.pem"
LOG_FILE="/home/brad/projects/robot-overlord-news/upload.log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

rsync -avz --progress -e "ssh -i $SSH_KEY" "$DB_SOURCE" "$DB_DEST" >> "$LOG_FILE" 2>&1

ssh -i "$SSH_KEY" ec2-user@44.221.25.226 'sudo systemctl restart news' >> "$LOG_FILE" 2>&1

echo "Done." >> "$LOG_FILE"
