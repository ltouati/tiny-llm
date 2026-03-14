#!/bin/bash
set -e

INSTANCE_NAME="tiny-llm-tokenizer"
ZONE="us-central1-f"
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="gs://tiny-llm-workspace-${PROJECT_ID}"
# Using a 56 Core compute-optimized machine to fit within the 100 GCP CPU quota!
MACHINE_TYPE="c2d-highcpu-56"

function handle_ctrlc() {
    echo -e "\n🛑 Caught CTRL-C! Deleting the VM..."
    rm -f source_tokenizer.tar.gz startup-tokenizer.sh target.tar.gz
    
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
        echo "Deleting VM $INSTANCE_NAME..."
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet || true
        echo "✅ VM deleted."
    fi
    exit 130
}

trap handle_ctrlc INT

echo "🚀 Step 1: Preparing GCS Bucket ($BUCKET_NAME)..."
if ! gcloud storage ls "$BUCKET_NAME" &> /dev/null; then
    echo "Creating bucket..."
    gcloud storage buckets create "$BUCKET_NAME" --location=us-central1
fi

echo "📦 Step 2: Archiving locally-modified Rust source code..."
tar -czf source_tokenizer.tar.gz src/ Cargo.toml Cargo.lock build.rs config.json tokenizer.json .agent/

echo "☁️  Step 3: Uploading assets to GCP Storage..."
gcloud storage cp source_tokenizer.tar.gz "$BUCKET_NAME/run/"

# Create the startup script locally
cat << 'EOF' > startup-tokenizer.sh
#!/bin/bash
set -e

# Log everything to a file
exec > >(tee -a /var/log/startup-tokenizer.log) 2>&1

echo "Starting headless TinyLLM 10B Tokenization pipeline on 112-Core GCP Instance..."

export HOME=/root
export PATH="$HOME/.cargo/bin:$PATH"

# Install dependencies if not present
if ! command -v rustc &> /dev/null; then
    echo "Installing system dependencies..."
    apt-get update -y
    apt-get install -y build-essential curl wget pkg-config libssl-dev
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

mkdir -p /opt/tiny-llm
cd /opt/tiny-llm

BUCKET_NAME="[INJECT_BUCKET_NAME]"
INSTANCE_NAME="[INJECT_INSTANCE_NAME]"
ZONE="[INJECT_ZONE]"

echo "Downloading source payload from GCS..."
gcloud storage cp "$BUCKET_NAME/run/source_tokenizer.tar.gz" .
tar -xzf source_tokenizer.tar.gz
rm source_tokenizer.tar.gz

echo "Compiling TinyLLM tokenizer (release profile)..."
cargo build --release --bin prep_data

echo "⚡ Launching Ultra-Fast Native tokenization across AMD EPYC Cores..."
./target/release/prep_data || { echo "❌ Tokenization crashed or was terminated. Cleanup initiated."; }

echo "☁️ Uploading 40GB final fineweb_edu.bin strictly to Bucket cache..."
gcloud storage cp fineweb_edu.bin "$BUCKET_NAME/cache/" || true

echo "Preparation complete! Self-destructing VM to save costs..."
gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet

EOF

# Inject variables
sed -i "s|\[INJECT_BUCKET_NAME\]|$BUCKET_NAME|g" startup-tokenizer.sh
sed -i "s|\[INJECT_INSTANCE_NAME\]|$INSTANCE_NAME|g" startup-tokenizer.sh
sed -i "s|\[INJECT_ZONE\]|$ZONE|g" startup-tokenizer.sh

echo "🌩️ Step 4: Fire-and-Forget VM Provisioning ($MACHINE_TYPE)..."
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo "❌ VM '$INSTANCE_NAME' already exists! Deleting it to refresh..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone=$ZONE --quiet
fi

# Preemptible flag is used heavily for cost savings since this is an ephemeral data pipeline
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --preemptible \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=ai-frontend-network \
    --metadata-from-file=startup-script=startup-tokenizer.sh \
    --metadata=enable-osconfig=TRUE,enable-oslogin=true \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --service-account=231108083037-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2404-noble-amd64-v20240423,mode=rw,size=200,type=pd-balanced \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ops-agent-policy=v2-template-1-5-0,goog-ec-src=vm_add-gcloud

echo "✅ Compute VM launched successfully!"
echo "📡 Establishing SSH connection to stream tokenization logs..."

rm source_tokenizer.tar.gz startup-tokenizer.sh

# Watch SSH
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="echo 'SSH is up'" &> /dev/null; then
        echo "✅ Connected to Tokenizer SSH daemon!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "Waiting for VM SSH daemon to initialize... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 10
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Failed to connect to SSH. Check VM instance logs in Google Cloud Console."
    exit 1
fi

echo "Streaming logs now:"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="tail -f /var/log/startup-tokenizer.log" || true

echo "🛑 Log stream disconnected (VM likely self-destructed successfully)."
