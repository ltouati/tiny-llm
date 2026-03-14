---
name: Remote Evaluator
description: Connects to an actively running GCP training VM, downloads the latest checkpoint snapshot to the local environment, and runs the HellaSwag evaluation suite on it.
---

# Remote Checkpoint Evaluation

This skill allows you to seamlessly evaluate the TinyLLM on your local machine while it is still actively training in the cloud, without pausing the pipeline.

## How to use this skill

1. **Run the Retrieval & Evaluation**:
   - The helper script `scripts/run_remote_eval.sh` handles the end-to-end process.
   - Example: `bash .agent/skills/remote_evaluation/scripts/run_remote_eval.sh`

2. **What it Does**:
   - The script connects to the `a100-trainer` Compute Engine instance over SSH.
   - It lists all `fineweb_checkpoint_*.safetensors` files inside the running `/opt/tiny-llm/` directory and targets the newest timestamp.
   - It uses `gcloud compute scp` to explicitly download that single large snapshot down to the local project root.
   - It then hands off the checkpoint to the `Evaluator` skill via `.agent/skills/evaluation/scripts/run_eval.sh` to compile native bindings, memory-map the new weights onto the local GPU, and benchmark it against the 10,042 HellaSwag questions.
   - Finally, it passes the weights to the `Generator` skill via `.agent/skills/generation/scripts/test_generation.sh` to output an AI text completion sample, verifying qualitative reasoning gains.
