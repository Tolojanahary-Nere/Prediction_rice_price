#!/bin/bash
export COMPOSE_HTTP_TIMEOUT=300
docker-compose down -v
docker system prune -af
docker-compose up --build
