#!/bin/sh
echo "ARM v7 FastAPI Base Image Build Script"

# 設定変数
DOCKER_USERNAME="mdottdda99"
IMAGE_NAME="mdot99"
IMAGE_TAG="JP2PROMPT"

convert favicon.ico.svg -resize 32x32 favicon.ico

echo
echo "===== Docker Hub ログイン ====="
docker login

echo
echo "===== Buildx builderの作成/使用 ====="
docker buildx create --name arm-builder --use 2>/dev/null || docker buildx use arm-builder

echo
echo "===== ARM v7 イメージをビルドしてプッシュ ====="
docker buildx build \
  --platform linux/arm/v7 \
  --tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} \
  --push \
  .

echo
echo "===== ビルド完了 ====="
echo "イメージ: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "プラットフォーム: linux/arm/v7"
echo "DockerHub URL: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"