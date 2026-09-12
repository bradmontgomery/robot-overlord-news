# Deployment Guide

Single EC2 instance running the Go web app + the news-fetching agent side by side,
sharing a SQLite database at `/opt/news/data/news.db`.

## Architecture

```
Internet → Elastic IP → Nginx (:80/:443) → news_site binary (:8080)
                                           ↕ reads
                                        /opt/news/data/news.db
                                           ↑ writes
                                        news-fetching agent (cron / systemd)
```

## Prerequisites

- AWS CLI configured (`aws configure`) with a profile that can create EC2 resources
- An EC2 key pair in the target region (`aws ec2 describe-key-pairs`)
- A domain name with an A record you can update (Cloudflare, Namecheap, Route 53, etc.)
- `rsync` and `ssh` available locally

## First-Time Setup

### 1. Find the right AMI for your region

The CloudFormation template defaults to us-east-1. For other regions:

```bash
aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
  --output text
```

Update `ImageId` in `infrastructure/cloudformation.yaml` if needed.

### 2. Deploy the CloudFormation stack

```bash
make infra-deploy \
  KEY_PAIR=$KEY_PAIR \
  SSH_CIDR=$(curl -s https://checkip.amazonaws.com)/32 \
  DOMAIN=$DOMAIN
```

Note the `PublicIP` from the stack outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name robot-overlord-news \
  --query 'Stacks[0].Outputs'
```

### 3. Point your domain at the Elastic IP

Add an A record: `robot-overlord.news → <PublicIP from step 2>`

DNS propagation takes a few minutes to hours depending on your provider.

### 4. Do the first deploy

```bash
make deploy DEPLOY_HOST=<PublicIP>
```

The systemd service will start the Go app. Verify it's running:

```bash
ssh ec2-user@<PublicIP> 'sudo systemctl status news'
```

### 5. Set up TLS with Let's Encrypt

Wait for DNS to propagate first, then:

```bash
ssh ec2-user@<PublicIP>
sudo certbot --nginx -d robot-overlord.news
```

Certbot will modify `/etc/nginx/conf.d/news.conf` to add HTTPS and an HTTP→HTTPS redirect.
It sets up a renewal *service*, but **not a working timer** — see below.

**Enable the renewal timer. It is not automatic.** On Amazon Linux 2023 the certbot package
ships `certbot-renew.timer` **disabled**, so nothing ever renews and the cert silently expires
90 days later. This took the site down on 2026-09-12:

```bash
sudo systemctl enable --now certbot-renew.timer
systemctl list-timers '*certbot*'
```

`NEXT` and `LEFT` must be populated. `0 timers listed` means it did not take.

The timer fires daily; certbot only acts once the cert is inside its 30-day renewal window.

Verify HTTPS is working:

```bash
curl https://robot-overlord.news
```

## Deploying Updates

Deployment is automated via GitHub Actions. Any push to `main` (including a merged PR)
triggers the CI/CD pipeline defined in `.github/workflows/deploy.yml`, which:

1. Runs `go test ./...`
2. Builds the binary with `go build`
3. Rsyncs the binary to `/opt/news/` on the EC2 instance
4. Restarts the `news` systemd service

### Required GitHub Secrets

Add these under **Settings → Secrets and variables → Actions** in the GitHub repo:

| Secret | Value |
|--------|-------|
| `SSH_PRIVATE_KEY` | Full contents of `~/.ssh/robot-overlord.pem` |
| `DEPLOY_HOST` | `robot-overlord.news` |
| `DEPLOY_USER` | `ec2-user` |

### Manual Deploy

If you need to deploy outside of CI (e.g., for a hotfix without a PR):

```bash
make deploy DEPLOY_HOST=robot-overlord.news
```

## Deploying the News-Fetching Agent

The agent that populates the database runs on the same EC2 instance. Deploy it however
makes sense for that project (likely a Python venv + cron job or its own systemd service).

The agent should write to `/opt/news/data/news.db`. The Go web app opens this file
read-only with WAL mode, so concurrent reads during a write are safe.

Example cron entry to run the agent hourly (edit with `crontab -e` as the `ec2-user`):

```cron
0 * * * * /home/ec2-user/langchain-agent/run.sh >> /var/log/news-agent.log 2>&1
```

## Useful Commands

```bash
# SSH into the instance
ssh ec2-user@robot-overlord.news

# View web app logs (live)
sudo journalctl -u news -f

# View Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Restart the web app
sudo systemctl restart news

# Check Nginx config is valid before reloading
sudo nginx -t && sudo systemctl reload nginx

# Check TLS cert expiry
sudo certbot certificates

# Check the database has data
sqlite3 /opt/news/data/news.db "SELECT COUNT(*), MAX(fetched_at) FROM news_results;"
```

## Updating the CloudFormation Stack

To change instance type or SSH CIDR:

```bash
make infra-deploy KEY_PAIR=your-key INSTANCE_TYPE=t3.small SSH_CIDR=1.2.3.4/32
```

**Note:** Changing `InstanceType` will stop and restart the EC2 instance (brief downtime).
The Elastic IP stays attached, so your domain continues to work.

## Tearing Down

```bash
aws cloudformation delete-stack --stack-name robot-overlord-news
```

This deletes the EC2 instance, security group, and Elastic IP. The database is gone too
unless you've backed it up.

## Cost

| Resource | Cost |
|----------|------|
| t3.micro EC2 | ~$8.50/mo (free tier: first 12 months) |
| 8 GB gp3 EBS | ~$0.64/mo |
| Elastic IP (running) | Free |
| Elastic IP (stopped) | $0.005/hr |
| Data transfer | Negligible for personal use |
| **Total** | **~$9–10/mo** |
