#!/bin/sh

docker run --name redis-stack -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
#docker run --name redis-server -d -p 6379:6379 redis:latest