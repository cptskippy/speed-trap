#!/bin/bash

set -e

COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"

echo "Running Docker Compose from $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up --build
#docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up
