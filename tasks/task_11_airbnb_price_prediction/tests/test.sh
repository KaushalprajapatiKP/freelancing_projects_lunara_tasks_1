#!/bin/bash

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p /logs/verifier

write_reward() {
    echo "$1" > "$REWARD_FILE"
}

trap 'if [ $? -ne 0 ]; then write_reward 0; fi' EXIT

apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh

source $HOME/.local/bin/env

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

timeout 7200 uvx \
  -p 3.12 \
  -w pytest==8.4.1 \
  -w pandas==2.2.0 \
  -w numpy==1.26.4 \
  pytest /tests/test_outputs.py -rA

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

