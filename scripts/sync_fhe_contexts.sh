#!/bin/bash
# Helper script to sync FHE context files to all client devices
# This must be run before executing DIWS-FHE experiments in distributed mode
#
# Usage:
#   ./scripts/sync_fhe_contexts.sh
#
# This script will:
#   1. Generate FHE context files on the server (if not present)
#   2. Distribute client_context.pkl to all client devices

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if ansible is available
if ! command -v ansible-playbook &> /dev/null; then
    echo "Error: ansible-playbook not found. Please install ansible."
    exit 1
fi

# Check if inventory exists
INVENTORY_PATH="$PROJECT_ROOT/deployment/ansible/inventory.yml"
if [ ! -f "$INVENTORY_PATH" ]; then
    echo "Error: Inventory file not found at $INVENTORY_PATH"
    exit 1
fi

# Check if sync_fhe_context.yml exists
PLAYBOOK_PATH="$PROJECT_ROOT/deployment/ansible/sync_fhe_context.yml"
if [ ! -f "$PLAYBOOK_PATH" ]; then
    echo "Error: Playbook not found at $PLAYBOOK_PATH"
    exit 1
fi

echo "=================================="
echo "Syncing FHE Context Files"
echo "=================================="
echo ""
echo "This will:"
echo "  1. Generate FHE context files on the server (if needed)"
echo "  2. Distribute client_context.pkl to all client devices"
echo ""

# Run the ansible playbook
cd "$PROJECT_ROOT"
ansible-playbook "$PLAYBOOK_PATH" -i "$INVENTORY_PATH"

echo ""
echo "=================================="
echo "FHE context sync completed!"
echo "=================================="
echo ""
echo "You can now run DIWS-FHE experiments on the hardware testbed."
