#!/bin/bash
set -e

INSTANCE_NAME="a100-benchmark"
ZONE="us-central1-f"
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="gs://tiny-llm-workspace-${PROJECT_ID}"
MACHINE_IMAGE="tiny-llm-a100-base"
RUN_ID=$(date +%Y%m%d_%H%M%S)

function handle_ctrlc() {
    echo -e "\n🛑 Caught CTRL-C! Cleaning up..."
    rm -f bench_source.tar.gz bench-startup-script.sh target.tar.gz
    
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
        echo "Deleting VM $INSTANCE_NAME..."
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
        echo "✅ VM deleted."
    else
        echo "VM $INSTANCE_NAME does not exist. Nothing to clean up."
    fi
    exit 130
}

trap handle_ctrlc INT

echo "🚀 Step 1: Preparing GCS Bucket ($BUCKET_NAME)..."
if ! gcloud storage ls "$BUCKET_NAME" &> /dev/null; then
    gcloud storage buckets create "$BUCKET_NAME" --location=us-central1
fi

echo "📦 Step 2: Compiling Rust tokenization and archiving source..."
cargo build --release --bin prep_data

if ! gcloud storage ls "$BUCKET_NAME/cache/fineweb_edu.bin" &> /dev/null; then
    echo "⚠️ Tokenized dataset not found in bucket. Generating 10 Billion tokens natively via local CPU..."
    ./target/release/prep_data
    gcloud storage cp fineweb_edu.bin "$BUCKET_NAME/cache/"
fi

tar -czf bench_source.tar.gz src/ Cargo.toml Cargo.lock build.rs config.json tokenizer.json .agent/

echo "☁️  Step 3: Uploading assets to GCP Storage..."
gcloud storage cp bench_source.tar.gz "$BUCKET_NAME/run/"

IMAGE_EXISTS=false
if gcloud compute machine-images describe $MACHINE_IMAGE &> /dev/null; then
    echo "✅ Found custom Machine Image '$MACHINE_IMAGE'. Will deploy from it to save time!"
    IMAGE_EXISTS=true
    DISK_ARG="--source-machine-image=$MACHINE_IMAGE"
else
    echo "⚠️ Machine image '$MACHINE_IMAGE' not found. Will provision from standard base image."
    DISK_ARG="--create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225,mode=rw,size=200,type=pd-balanced"
fi

cat << 'INNSCRIPT' > bench-startup-script.sh
#!/bin/bash
set -e
exec > >(tee -a /var/log/startup-training.log) 2>&1

echo "Starting headless TinyLLM Benchmarking..."

while ! command -v nvidia-smi &> /dev/null; do
    sleep 15
done

export HOME=/root
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"

if ! command -v rustc &> /dev/null; then
    apt-get update -y
    apt-get install -y build-essential curl wget pkg-config libssl-dev nvidia-cuda-toolkit
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

mkdir -p /opt/tiny-llm
cd /opt/tiny-llm

BUCKET_NAME="[INJECT_BUCKET_NAME]"
INSTANCE_NAME="[INJECT_INSTANCE_NAME]"
ZONE="[INJECT_ZONE]"
RUN_ID="[INJECT_RUN_ID]"

gcloud storage cp "$BUCKET_NAME/run/bench_source.tar.gz" .
tar -xzf bench_source.tar.gz
rm bench_source.tar.gz

gcloud storage cp "$BUCKET_NAME/cache/fineweb_edu.bin" . || true

gcloud storage cp "$BUCKET_NAME/cache/target.tar.gz" . || true
if [ -f "target.tar.gz" ]; then
    tar -xzf target.tar.gz
    rm target.tar.gz
fi

cargo build --release --bin train_benchmark
tar -czf target.tar.gz target/
gcloud storage cp target.tar.gz "$BUCKET_NAME/cache/" || true

echo "Running benchmark loop..."
MAX_STEPS=500 ./target/release/train_benchmark || true

echo "Benchmark complete."
gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet

INNSCRIPT

sed -i "s|\[INJECT_BUCKET_NAME\]|$BUCKET_NAME|g" bench-startup-script.sh
sed -i "s|\[INJECT_INSTANCE_NAME\]|$INSTANCE_NAME|g" bench-startup-script.sh
sed -i "s|\[INJECT_ZONE\]|$ZONE|g" bench-startup-script.sh
sed -i "s|\[INJECT_RUN_ID\]|$RUN_ID|g" bench-startup-script.sh

echo "🌩️ Step 4: Provisioning Benchmark VM..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=ai-frontend-network \
    --metadata-from-file=startup-script=bench-startup-script.sh \
    --metadata=enable-osconfig=TRUE,enable-oslogin=true \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --service-account=231108083037-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    $DISK_ARG 

echo "✅ VM launched successfully!"
rm bench_source.tar.gz bench-startup-script.sh

echo "Waiting for completion... You can track logs or just wait for it to delete itself."
