BINARY      := news_site
MODULE      := $(shell go list -m)
STACK_NAME  := robot-overlord-news
DEPLOY_USER ?= ec2-user
DEPLOY_HOST ?= $(error Set DEPLOY_HOST=<host-or-ip>)
SSH_KEY     ?= ~/.ssh/robot-overlord.pem
INSTANCE_TYPE ?= t3.micro
SSH_CIDR    ?= $(error Set SSH_CIDR=<your-ip>/32)
DOMAIN      ?= ""

.PHONY: all build run clean test fmt vet lint tidy deploy infra-deploy

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
	  $(BINARY) templates/ static/ \
	  $(DEPLOY_USER)@$(DEPLOY_HOST):/opt/news/
	ssh -i $(SSH_KEY) $(DEPLOY_USER)@$(DEPLOY_HOST) 'sudo systemctl restart news && sudo systemctl status news --no-pager'

# Provision AWS infrastructure via CloudFormation
# Usage: make infra-deploy KEY_PAIR=mykey SSH_CIDR=1.2.3.4/32 DOMAIN=news.example.com
infra-deploy:
	aws cloudformation deploy \
	  --template-file infrastructure/cloudformation.yaml \
	  --stack-name $(STACK_NAME) \
	  --parameter-overrides \
	    KeyPairName=$(KEY_PAIR) \
	    InstanceType=$(INSTANCE_TYPE) \
	    AllowedSSHCidr=$(SSH_CIDR) \
	    DomainName=$(DOMAIN) \
	  --capabilities CAPABILITY_IAM
	@echo ""
	@echo "Stack outputs:"
	@aws cloudformation describe-stacks \
	  --stack-name $(STACK_NAME) \
	  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
	  --output table
