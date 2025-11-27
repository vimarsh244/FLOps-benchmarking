#!/bin/bash
# simple ssh-based deployment script for flops benchmarking
# usage: ./deploy.sh <inventory_file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INVENTORY_FILE="${1:-$SCRIPT_DIR/../ansible/inventory.yml}"

# parse inventory file (simplified yaml parsing)
get_hosts() {
    grep -E "^\s+\w+:" "$INVENTORY_FILE" | grep -v "vars:" | grep -v "hosts:" | grep -v "children:" | sed 's/://g' | awk '{print $1}'
}

get_host_var() {
    local host=$1
    local var=$2
    grep -A10 "^[[:space:]]*${host}:" "$INVENTORY_FILE" | grep "${var}:" | head -1 | sed "s/.*${var}:[[:space:]]*//" | tr -d '"'
}

echo "FLOps Benchmarking Deployment Script"
echo "====================================="
echo "Repository: $REPO_DIR"
echo "Inventory: $INVENTORY_FILE"
echo ""

# sync code to all hosts
echo "Syncing code to all devices..."

for host in $(get_hosts); do
    ansible_host=$(get_host_var "$host" "ansible_host")
    ansible_user=$(get_host_var "$host" "ansible_user")
    
    if [ -n "$ansible_host" ] && [ -n "$ansible_user" ]; then
        echo "  Syncing to $host ($ansible_user@$ansible_host)..."
        
        rsync -avz --delete \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='outputs/' \
            --exclude='.venv' \
            --exclude='*.egg-info' \
            -e "ssh -o StrictHostKeyChecking=no" \
            "$REPO_DIR/" "$ansible_user@$ansible_host:~/flops-benchmarking/" || {
                echo "  Warning: Failed to sync to $host"
            }
    fi
done

echo ""
echo "Deployment complete!"
echo ""
echo "To run an experiment:"
echo "  cd $REPO_DIR"
echo "  ansible-playbook deployment/ansible/run_experiment.yml -i deployment/ansible/inventory.yml"

