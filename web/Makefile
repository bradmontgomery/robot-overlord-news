BINARY      := news_site
DEPLOY_USER ?= ec2-user
DEPLOY_HOST ?= $(error Set DEPLOY_HOST=<host-or-ip>)
SSH_KEY     ?= ~/.ssh/robot-overlord.pem

.PHONY: all build run clean test fmt vet tidy deploy

all: build

build:
	go build -o $(BINARY) .

run: build
	./$(BINARY)

test:
	go test ./...

fmt:
	gofmt -w .

vet:
	go vet ./...

tidy:
	go mod tidy

clean:
	rm -f $(BINARY)

# Deploy app to EC2: build binary, sync files, restart service
# Usage: make deploy DEPLOY_HOST=news.yourdomain.com
deploy: build
	rsync -avz --progress -e "ssh -i $(SSH_KEY)" \
	  $(BINARY) \
	  $(DEPLOY_USER)@$(DEPLOY_HOST):/opt/news/
	ssh -i $(SSH_KEY) $(DEPLOY_USER)@$(DEPLOY_HOST) 'sudo systemctl restart news && sudo systemctl status news --no-pager'
