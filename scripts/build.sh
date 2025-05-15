#!/bin/bash

set -e  # exit on any error

APP_NAME="speed-trap"
DOCKERFILE_PATH="docker/Dockerfile"
BUILD_CONTEXT="."
TAG="latest"

echo "🔨 Building Docker image $APP_NAME:$TAG..."
docker build -f $DOCKERFILE_PATH -t $APP_NAME:$TAG $BUILD_CONTEXT
echo "✅ Done!"