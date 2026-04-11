#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MANIFEST="${ROOT_DIR}/scripts/manifests/personalized_fast_seed0_core.csv"

MANIFEST_PATH="${DEFAULT_MANIFEST}"
ONLY_STRATEGY=""
LIMIT=0
START_AT=1
DRY_RUN=false
NUM_ROUNDS=200

usage() {
  cat <<'EOF'
Run fast personalized FL experiments (FedPer + Ditto, single seed).

Usage:
  scripts/run_personalized_fast.sh [options]

Options:
  --manifest <path>        Path to manifest CSV (default: scripts/manifests/personalized_fast_seed0_core.csv)
  --strategy <name>        Filter by strategy: fedper or ditto
  --start-at <index>       Start from 1-based run index after filtering (default: 1)
  --limit <n>              Run at most n experiments after --start-at (default: 0 = no limit)
  --rounds <n>             Server training rounds override (default: 200)
  --dry-run                Print commands only
  -h, --help               Show this help

Examples:
  scripts/run_personalized_fast.sh --dry-run
  scripts/run_personalized_fast.sh --strategy ditto --limit 3
  scripts/run_personalized_fast.sh --start-at 10
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --strategy)
      ONLY_STRATEGY="$2"
      shift 2
      ;;
    --start-at)
      START_AT="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}"
  exit 1
fi

if [[ -n "${ONLY_STRATEGY}" && "${ONLY_STRATEGY}" != "fedper" && "${ONLY_STRATEGY}" != "ditto" ]]; then
  echo "Invalid strategy filter: ${ONLY_STRATEGY}. Use fedper or ditto."
  exit 1
fi

if ! [[ "${START_AT}" =~ ^[0-9]+$ ]] || [[ "${START_AT}" -lt 1 ]]; then
  echo "--start-at must be a positive integer"
  exit 1
fi

if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "--limit must be an integer >= 0"
  exit 1
fi

if ! [[ "${NUM_ROUNDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_ROUNDS}" -lt 1 ]]; then
  echo "--rounds must be a positive integer"
  exit 1
fi

echo "Using manifest: ${MANIFEST_PATH}"
echo "Project root: ${ROOT_DIR}"
echo "Strategy filter: ${ONLY_STRATEGY:-none}"
echo "Start at: ${START_AT}"
echo "Limit: ${LIMIT}"
echo "Server rounds: ${NUM_ROUNDS}"
echo "Dry run: ${DRY_RUN}"

attempted=0
ran=0
failed=0
eligible_index=0

while IFS=, read -r run_id enabled experiment_name strategy dataset model partitioner scenario seed extra_overrides; do
  if [[ "${run_id}" == "run_id" ]]; then
    continue
  fi

  if [[ "${enabled}" != "true" ]]; then
    continue
  fi

  if [[ -n "${ONLY_STRATEGY}" && "${strategy}" != "${ONLY_STRATEGY}" ]]; then
    continue
  fi

  eligible_index=$((eligible_index + 1))
  if [[ "${eligible_index}" -lt "${START_AT}" ]]; then
    continue
  fi

  if [[ "${LIMIT}" -gt 0 && "${ran}" -ge "${LIMIT}" ]]; then
    break
  fi

  attempted=$((attempted + 1))
  echo ""
  echo "================================================================================"
  echo "Run ${attempted}: ${experiment_name}"
  echo "  strategy=${strategy} dataset=${dataset} model=${model}"
  echo "  partitioner=${partitioner} scenario=${scenario} seed=${seed}"
  echo "================================================================================"

  cmd=(python -m src.main
    "strategy=${strategy}"
    "dataset=${dataset}"
    "model=${model}"
    "partitioner=${partitioner}"
    "scenario=${scenario}"
    "experiment.name=${experiment_name}"
    "experiment.seed=${seed}"
    "server.num_rounds=${NUM_ROUNDS}"
  )

  if [[ -n "${extra_overrides}" ]]; then
    read -r -a extra_parts <<< "${extra_overrides}"
    for override in "${extra_parts[@]}"; do
      cmd+=("${override}")
    done
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "${cmd[@]}"
    printf '\n'
    ran=$((ran + 1))
    continue
  fi

  set +e
  (
    cd "${ROOT_DIR}"
    "${cmd[@]}" < /dev/null
  )
  exit_code=$?
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "Run failed with exit code ${exit_code}: ${experiment_name}"
    failed=$((failed + 1))
  else
    ran=$((ran + 1))
  fi
done < "${MANIFEST_PATH}"

echo ""
echo "Completed. successful=${ran} failed=${failed} attempted=${attempted}"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
