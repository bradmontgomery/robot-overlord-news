---
name: robot-overlord-news-deploy
description: |
  Build and deploy the robot-overlord-news Go web app to EC2.
  Use when the user says things like: "deploy the app", "deploy the site",
  "ship it", "push the latest code", "run a deploy", "update the server",
  "deploy robot-overlord-news", or anything implying deploying the web app
  code/binary to production.
  Do NOT use for uploading the news database — use robot-overlord-news-deploy-db
  for that.
---

## Deploy robot-overlord-news App to EC2

### Step 1: Run make deploy

```bash
make deploy   # from the repo root
```

This single command:
- Builds the Go binary in `web/` (root Makefile delegates to `web/Makefile`)
- rsyncs the binary (templates and static assets are embedded) to `/opt/news/` on the server
- Restarts the `news` systemd service
- Prints service status

Relies on `DEPLOY_HOST` and `SSH_KEY` from the repo-root `.envrc` (loaded by direnv; see `.envrc.example`).

### Step 2: Check for errors

- If `make deploy` exits non-zero, surface the full error output and stop
- Common issues:
  - `DEPLOY_HOST` not set → run `direnv allow` in the project directory
  - rsync `Permission denied` → `/opt/news/` ownership issue on server; fix with `ssh -i "$SSH_KEY" "${DEPLOY_USER:-ec2-user}@$HOST_IP" 'sudo chown -R ec2-user:ec2-user /opt/news'`
  - `Host key verification failed` → stale known_hosts; fix with `ssh-keygen -R <host>`
  - DNS not resolving → override host: `make deploy DEPLOY_HOST=$HOST_IP`

### Step 3: Confirm success

Report success when the service status shows:
- `Active: active (running)`
- Log line: `listening on http://localhost:8080`
