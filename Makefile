BINARY := news_site
MODULE  := $(shell go list -m)

.PHONY: all build run clean test fmt vet lint tidy

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
