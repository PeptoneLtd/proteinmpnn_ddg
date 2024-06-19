.PHONY: all docker_image
all: docker_image push

REGISTRY := ghcr.io/peptoneltd
IMAGE := $(notdir $(CURDIR))
TAG := 4.0.0

docker_image:
	docker build --platform=linux/amd64 -t $(REGISTRY)/$(IMAGE):$(TAG) .
push:
	docker push $(REGISTRY)/$(IMAGE):$(TAG)