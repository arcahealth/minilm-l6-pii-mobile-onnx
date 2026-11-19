import os
import subprocess
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

OUTPUT_DIR = "./output_models"
RAW_DIR = "./raw_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

REPO_ID = "Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2"

print("=== 1. Downloading MiniLM-L6 PII Model & Tokenizer Assets ===")

# Download Tokenizer and Configs to OUTPUT_DIR
for fname in [
    "config.json",
    "tokenizer.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]:
    print(f"Downloading {fname} from {REPO_ID}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        local_dir=OUTPUT_DIR,
    )
    print(f"Saved {fname} to {OUTPUT_DIR}/{fname}")

# Download FP32 ONNX Model to RAW_DIR
print(f"Downloading onnx/model.onnx from {REPO_ID}...")
raw_model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="onnx/model.onnx",
    local_dir=RAW_DIR,
)
print(f"Downloaded raw model: {raw_model_path} ({os.path.getsize(raw_model_path) / 1e6:.2f} MB)")

print("\n=== 2. Running onnxslim Graph Simplification ===")
slim_model_path = os.path.join(OUTPUT_DIR, "minilm_l6_pii.slim.onnx")
print(f"Simplifying graph with onnxslim -> {slim_model_path}...")
subprocess.run(["onnxslim", raw_model_path, slim_model_path], check=True)
print(f"Simplified Model: {slim_model_path} ({os.path.getsize(slim_model_path) / 1e6:.2f} MB)")

print("\n=== 3. Running Dynamic INT8 Quantization ===")
int8_model_path = os.path.join(OUTPUT_DIR, "minilm_l6_pii.slim.int8.onnx")
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
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f" - {fname:<35} ({os.path.getsize(fpath) / 1e6:.2f} MB)")
