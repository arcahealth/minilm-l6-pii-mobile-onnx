import argparse
import os
import subprocess
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

OUTPUT_DIR = "./output_models"
RAW_DIR = "./raw_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

REPO_ID = "Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2"

parser = argparse.ArgumentParser(description="Download and optimize MiniLM-L6 PII ONNX models.")
parser.add_argument(
    "--force",
    action="store_true",
    help="Force re-downloading assets and re-optimizing models even if they already exist.",
)
args = parser.parse_args()

print("=== 1. Downloading MiniLM-L6 PII Model & Tokenizer Assets ===")

# Download Tokenizer and Configs to OUTPUT_DIR
for fname in [
    "config.json",
    "tokenizer.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]:
    dest = os.path.join(OUTPUT_DIR, fname)
    if not args.force and os.path.exists(dest):
        print(f"Asset {fname} already exists at {dest}, skipping download.")
    else:
        print(f"Downloading {fname} from {REPO_ID}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=OUTPUT_DIR,
        )
        print(f"Saved {fname} to {dest}")

# Download FP32 ONNX Model to RAW_DIR if needed
raw_model_path = os.path.join(RAW_DIR, "onnx", "model.onnx")
slim_model_path = os.path.join(OUTPUT_DIR, "minilm_l6_pii.slim.onnx")
int8_model_path = os.path.join(OUTPUT_DIR, "minilm_l6_pii.slim.int8.onnx")

needs_model_build = args.force or not os.path.exists(int8_model_path)
needs_raw_model = needs_model_build and not os.path.exists(slim_model_path)

if not args.force and os.path.exists(raw_model_path):
    print(f"Raw model already exists: {raw_model_path} ({os.path.getsize(raw_model_path) / 1e6:.2f} MB), skipping download.")
elif not needs_raw_model:
    if os.path.exists(int8_model_path):
        print(f"INT8 model already exists at {int8_model_path}, skipping raw model download.")
    else:
        print(f"Simplified model already exists at {slim_model_path}, skipping raw model download.")
else:
    print(f"Downloading onnx/model.onnx from {REPO_ID}...")
    raw_model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="onnx/model.onnx",
        local_dir=RAW_DIR,
    )
    print(f"Downloaded raw model: {raw_model_path} ({os.path.getsize(raw_model_path) / 1e6:.2f} MB)")

print("\n=== 2. Running onnxslim Graph Simplification ===")
if not args.force and os.path.exists(slim_model_path):
    print(f"Simplified model already exists: {slim_model_path} ({os.path.getsize(slim_model_path) / 1e6:.2f} MB), skipping onnxslim.")
elif not args.force and os.path.exists(int8_model_path):
    print(f"INT8 model already exists at {int8_model_path}, skipping onnxslim.")
else:
    print(f"Simplifying graph with onnxslim -> {slim_model_path}...")
    subprocess.run(["onnxslim", raw_model_path, slim_model_path], check=True)
    print(f"Simplified Model: {slim_model_path} ({os.path.getsize(slim_model_path) / 1e6:.2f} MB)")

print("\n=== 3. Running Dynamic INT8 Quantization ===")
if not args.force and os.path.exists(int8_model_path):
    print(f"INT8 quantized model already exists: {int8_model_path} ({os.path.getsize(int8_model_path) / 1e6:.2f} MB), skipping quantization.")
else:
    print(f"Quantizing model to INT8 -> {int8_model_path}...")
    quantize_dynamic(
        model_input=slim_model_path,
        model_output=int8_model_path,
        weight_type=QuantType.QUInt8,
        per_channel=True,
    )
    print(f"INT8 Quantized Model: {int8_model_path} ({os.path.getsize(int8_model_path) / 1e6:.2f} MB)")

print("\n=== Final Deployment Assets in ./output_models/ ===")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.startswith("."):
        continue
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f" - {fname:<35} ({os.path.getsize(fpath) / 1e6:.2f} MB)")
