#!/bin/bash
set -e

INSTANCE_NAME="a100-trainer"
ZONE="us-central1-f"

echo "🔍 Querying remote VM '$INSTANCE_NAME' for the latest checkpoint..."

# Get the latest checkpoint filename
LATEST_CHECKPOINT=$(gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="ls -t /opt/tiny-llm/*.safetensors | head -n 1" 2>/dev/null | tr -d '\r')

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoints found on the remote VM. Is the training running?"
    exit 1
fi

FILENAME=$(basename "$LATEST_CHECKPOINT")
echo "✅ Found latest checkpoint: $FILENAME"

echo "📥 Downloading $FILENAME from VM..."
gcloud compute scp "$INSTANCE_NAME:$LATEST_CHECKPOINT" . --zone="$ZONE" --quiet

echo "✅ Download complete! Passing to local Evaluator..."
echo "---------------------------------------------------"

# Trigger the existing local evaluation skill
bash .agent/skills/evaluation/scripts/run_eval.sh

echo ""
echo "---------------------------------------------------"
echo "✅ Evaluation complete! Triggering Generator to sample output..."

# Trigger the existing generation skill
bash .agent/skills/generation/scripts/test_generation.sh
