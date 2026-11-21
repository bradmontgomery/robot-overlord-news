# Thin delegator. Each component owns its own Makefile:
#   web/   — Go web app (make -C web <target>)
#   agent/ — Python news agent (make -C agent <target>)
STACK_NAME    := robot-overlord-news
INSTANCE_TYPE ?= t3.micro
SSH_CIDR      ?= $(shell curl -s https://checkip.amazonaws.com)/32
DOMAIN        ?= ""

.PHONY: all build run clean test fmt vet tidy deploy infra-deploy news report agent-test

all: build

# --- web ---
build run clean fmt vet tidy deploy:
	$(MAKE) -C web $@

# Both suites. web has no tests yet, so the agent suite is the one that matters.
test:
	$(MAKE) -C web test
	$(MAKE) -C agent test

# --- agent ---
news report:
	$(MAKE) -C agent $@

agent-test:
	$(MAKE) -C agent test

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
