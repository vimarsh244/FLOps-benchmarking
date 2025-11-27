#!/bin/bash
# run fl client on remote device
# usage: ./run_client.sh <host> <user> <partition_id> <server_address>

set -e

HOST="${1:?Host required}"
USER="${2:?User required}"
PARTITION_ID="${3:?Partition ID required}"
SERVER_ADDRESS="${4:-localhost:8080}"
SSH_KEY="${5:-~/.ssh/id_rsa}"

REMOTE_PATH="~/flops-benchmarking"
PYTHON_ENV="~/envs/flops/bin/python"

echo "Starting FL client on $USER@$HOST (partition $PARTITION_ID)"
echo "Server: $SERVER_ADDRESS"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$USER@$HOST" << EOF
    cd $REMOTE_PATH
    $PYTHON_ENV -m src.run_distributed client \
        --server-address $SERVER_ADDRESS \
        --partition-id $PARTITION_ID
EOF

echo "Client on $HOST completed"

