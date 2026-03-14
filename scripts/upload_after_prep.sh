#!/bin/bash
echo "Waiting for pre-existing prep_data task to complete..."
while pgrep -f "prep_data" > /dev/null; do
    sleep 300
done
echo "prep_data finished! Uploading the 40GB fineweb_edu.bin dataset into the GCS Cache..."
PROJECT_ID=$(gcloud config get-value project)
gcloud storage cp fineweb_edu.bin gs://tiny-llm-workspace-${PROJECT_ID}/cache/
echo "✅ Cached!"
