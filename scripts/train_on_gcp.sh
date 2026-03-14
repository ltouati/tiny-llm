#!/bin/bash
set -e

INSTANCE_NAME="a100-trainer"
ZONE="us-central1-f"
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="gs://tiny-llm-workspace-${PROJECT_ID}"
MACHINE_IMAGE="tiny-llm-a100-base"
RUN_ID=$(date +%Y%m%d_%H%M%S)

function handle_ctrlc() {
    echo -e "\n🛑 Caught CTRL-C! Saving checkpoints and deleting the VM..."
    rm -f source.tar.gz startup-script.sh target.tar.gz
    
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
        echo "Attempting to upload checkpoints from VM..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --ssh-flag="-o ConnectTimeout=10" --command="cd /opt/tiny-llm && gcloud storage cp *.safetensors $BUCKET_NAME/checkpoints/$RUN_ID/" 2>/dev/null || echo "⚠️ No checkpoints found or SSH not ready."
        
        if [ "$IMAGE_EXISTS" = "false" ]; then
            echo "Removing safetensors before baking machine image..."
            gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --ssh-flag="-o ConnectTimeout=10" --command="rm -f /opt/tiny-llm/*.safetensors" 2>/dev/null || true
            
            echo "Shutting down VM so host can bake machine image..."
            gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --ssh-flag="-o ConnectTimeout=10" --command="sudo sync && sudo poweroff" 2>/dev/null || true
            
            echo "Waiting 30 seconds for the VM to fully spin down..."
            sleep 30
            
            echo "Baking custom machine image $MACHINE_IMAGE..."
            gcloud compute machine-images create "$MACHINE_IMAGE" \
                --source-instance="$INSTANCE_NAME" \
                --source-instance-zone="$ZONE" \
                --quiet || echo "⚠️ Failed to bake machine image."
        fi
        
        echo "Deleting VM $INSTANCE_NAME..."
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
        echo "✅ VM deleted."
        
        echo ""
        echo "⬇️  To download your checkpoints locally, run:"
        echo "    mkdir -p checkpoints/$RUN_ID"
        echo "    gcloud storage cp $BUCKET_NAME/checkpoints/$RUN_ID/*.safetensors checkpoints/$RUN_ID/"
    else
        echo "VM $INSTANCE_NAME does not exist. Nothing to clean up."
    fi
    exit 130
}

trap handle_ctrlc INT

echo "🚀 Step 1: Preparing GCS Bucket ($BUCKET_NAME)..."
if ! gcloud storage ls "$BUCKET_NAME" &> /dev/null; then
    echo "Creating bucket..."
    gcloud storage buckets create "$BUCKET_NAME" --location=us-central1
fi

echo "🔐 Step 1.5: Granting IAM permissions to the Compute Service Account..."
gcloud storage buckets add-iam-policy-binding "$BUCKET_NAME" \
    --member="serviceAccount:231108083037-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin" \
    --quiet

echo "🔑 Step 1.8: Securing HuggingFace Token..."
if [ -f ".env" ]; then
    source .env
fi
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set in .env"
    exit 1
fi
echo "$HF_TOKEN" > hf_token.tmp
gcloud storage cp hf_token.tmp "$BUCKET_NAME/secrets/hf_token.txt"
rm hf_token.tmp

echo "📦 Step 2: Compiling Rust tokenization and archiving source..."
cargo build --release --bin prep_data

if ! gcloud storage ls "$BUCKET_NAME/cache/fineweb_edu.bin" &> /dev/null; then
    echo "⚠️ Tokenized dataset not found in bucket. Generating 10 Billion tokens natively via local CPU..."
    ./target/release/prep_data
    echo "☁️ Uploading fineweb_edu.bin to Bucket cache..."
    gcloud storage cp fineweb_edu.bin "$BUCKET_NAME/cache/"
else
    echo "✅ Found a cached tokenization payload in bucket. Skipping 10B generation."
fi

tar -czf source.tar.gz src/ Cargo.toml Cargo.lock build.rs config.json tokenizer.json .agent/

echo "☁️  Step 3: Uploading assets to GCP Storage..."
gcloud storage cp source.tar.gz "$BUCKET_NAME/run/"

# Check if Machine Image exists
IMAGE_EXISTS=false
if gcloud compute machine-images describe $MACHINE_IMAGE &> /dev/null; then
    echo "✅ Found custom Machine Image '$MACHINE_IMAGE'. Will deploy from it to save time!"
    IMAGE_EXISTS=true
    DISK_ARG="--source-machine-image=$MACHINE_IMAGE"
else
    echo "⚠️ Machine image '$MACHINE_IMAGE' not found. Will provision from standard base image and bake it later."
    DISK_ARG="--create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225,mode=rw,size=200,type=pd-balanced"
fi

# Create the startup script locally
cat << 'EOF' > startup-script.sh
#!/bin/bash
set -e

# Log everything to a file
exec > >(tee -a /var/log/startup-training.log) 2>&1

echo "Starting headless TinyLLM training pipeline..."

# Wait for NVIDIA drivers to come online (from Deep Learning Image)
while ! command -v nvidia-smi &> /dev/null; do
    echo "Waiting for NVIDIA drivers to initialize..."
    sleep 15
done
nvidia-smi

export HOME=/root
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Conditionally install dependencies
if ! command -v rustc &> /dev/null; then
    echo "Installing system dependencies..."
    apt-get update -y
    apt-get install -y build-essential curl wget pkg-config libssl-dev nvidia-cuda-toolkit

    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

echo "Downloading assets from GCS..."
mkdir -p /opt/tiny-llm
cd /opt/tiny-llm

# Retrieve variables passed via sed injection
BUCKET_NAME="[INJECT_BUCKET_NAME]"
INSTANCE_NAME="[INJECT_INSTANCE_NAME]"
ZONE="[INJECT_ZONE]"
IMAGE_EXISTS="[INJECT_IMAGE_EXISTS]"
RUN_ID="[INJECT_RUN_ID]"

gcloud storage cp "$BUCKET_NAME/run/source.tar.gz" .

tar -xzf source.tar.gz
rm source.tar.gz

echo "Checking if 10B tokenized payload is cached..."
gcloud storage cp "$BUCKET_NAME/cache/fineweb_edu.bin" . || echo "❌ Failed to pull fineweb_edu.bin cache. Check upload stage!"

echo "Attempting to pull the Cargo compilation cache..."
gcloud storage cp "$BUCKET_NAME/cache/target.tar.gz" . || echo "No existing cache found. Proceeding with a full fresh build."
if [ -f "target.tar.gz" ]; then
    echo "Restoring previous compilation cache..."
    tar -xzf target.tar.gz
    rm target.tar.gz
fi

echo "Compiling TinyLLM natively on A100..."
cargo build --release --bin train
echo "Archiving and uploading compilation cache back to Bucket..."
tar -czf target.tar.gz target/
gcloud storage cp target.tar.gz "$BUCKET_NAME/cache/" || true

echo "Running training loop..."
./target/release/train || { echo "❌ Training loop crashed or was terminated. Proceeding to cleanup..."; }

echo "Uploading checkpoints back to bucket..."
gcloud storage cp *.safetensors "$BUCKET_NAME/checkpoints/$RUN_ID/" || true

echo "Training complete."
if [ "$IMAGE_EXISTS" = "false" ]; then
    echo "Shutting down VM so host can bake machine image..."
    # Ensure all caches stream properly before halting
    sync
    sudo poweroff
else
    echo "Self-destructing VM to save costs..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
fi

EOF

# Inject variables into the startup script
sed -i "s|\[INJECT_BUCKET_NAME\]|$BUCKET_NAME|g" startup-script.sh
sed -i "s|\[INJECT_INSTANCE_NAME\]|$INSTANCE_NAME|g" startup-script.sh
sed -i "s|\[INJECT_ZONE\]|$ZONE|g" startup-script.sh
sed -i "s|\[INJECT_IMAGE_EXISTS\]|$IMAGE_EXISTS|g" startup-script.sh
sed -i "s|\[INJECT_RUN_ID\]|$RUN_ID|g" startup-script.sh

echo "🌩️ Step 4: Fire-and-Forget VM Provisioning..."
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo "❌ VM '$INSTANCE_NAME' already exists! Please delete it first."
    exit 1
fi

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=ai-frontend-network \
    --metadata-from-file=startup-script=startup-script.sh \
    --metadata=enable-osconfig=TRUE,enable-oslogin=true \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --service-account=231108083037-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/compute,https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    $DISK_ARG \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ops-agent-policy=v2-template-1-5-0,goog-ec-src=vm_add-gcloud \
    --reservation-affinity=none

echo "✅ VM launched successfully!"
echo "📡 Establishing SSH connection to stream training logs (this may take 1-2 minutes while the VM boots)..."

# Cleanup local temporary files before blocking
rm source.tar.gz startup-script.sh

# Exponential backoff loop to wait for SSH to become available
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="echo 'SSH is up'" &> /dev/null; then
        echo "✅ Connected to VM SSH daemon!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "Waiting for VM SSH daemon to initialize... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 10
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Failed to connect to the VM after $MAX_RETRIES attempts. The VM is still running in the background."
    exit 1
fi

echo "Streaming logs now:"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="tail -f /var/log/startup-training.log" || true

echo "🛑 Log stream disconnected."

# Run the same cleanup logic as if we received an INT signal
handle_ctrlc
