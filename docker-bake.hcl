// Docker Bake設定ファイル (docker-bake.hcl)
variable "DOCKER_REGISTRY" {
  default = "docker.io"
}

variable "DOCKER_USERNAME" {
  default = "mdottdda99"
}

variable "IMAGE_NAME" {
  default = "mdot99"
}

variable "IMAGE_TAG" {
  default = "JP2PROMPT"
}

group "default" {
  targets = ["JP2PROMPT"]
}

target "JP2PROMPT" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/arm/v7"]
  
  tags = [
    "${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
  ]
  
  output = ["type=registry,push=true"]
  
  labels = {
    "org.opencontainers.image.title" = "JP2PROMPT"
    "org.opencontainers.image.description" = "ARM v7 Alpine Linux base image with FastAPI"
    "org.opencontainers.image.source" = "https://github.com/mdottdda99/JP2PROMPT"
    "org.opencontainers.image.created" = timestamp()
  }
}