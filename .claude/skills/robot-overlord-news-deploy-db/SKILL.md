---
name: robot-overlord-news-deploy-db
description: |
  Upload a local SQLite news database to the robot-overlord-news EC2 server.
  Use when the user says things like: "deploy the db", "upload the database",
  "ship the db", "sync the news database", "update the db on the server",
  "push news_data.db", or anything about copying the news database to production.
  ALWAYS prompt for the source file path — never assume it, even if the user
  provides a path in their message, confirm it first.
  Do NOT use for deploying the web app binary or code changes — use
  robot-overlord-news-deploy for that.
---

## Deploy News Database to EC2

### Step 1: Always prompt for the source path

Use `AskUserQuestion` to ask for the local path to the database file. Always do this, even if the user mentioned a path in their message — confirm it.

### Step 2: Verify the file exists

```bash
ls -lh <path-from-user>
```

If the file doesn't exist, report the error and stop.

### Step 3: Upload with rsync

```bash
rsync -avz --progress -e "ssh -i ~/.ssh/robot-overlord.pem" \
  <local-path> ec2-user@${HOST_IP:-44.221.25.226}:/opt/news/data/news.db
```

### Step 4: Restart the service and verify

```bash
ssh -i ~/.ssh/robot-overlord.pem ec2-user@${HOST_IP:-44.221.25.226} \
  'sudo systemctl restart news && sudo systemctl status news --no-pager'
```

### Step 5: Report result

- Confirm the service shows `active (running)`
- Confirm the log line says `listening on http://localhost:8080` with no database warning
- If there's a warning like `unable to open database file`, report it and check that the upload succeeded
