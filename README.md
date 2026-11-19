# MiniLM-L6 PII Detection - Mobile ONNX Models

Ultra-compact, high-speed ONNX Runtime deployment assets for **MiniLM-L6** fine-tuned for end-to-end **Personally Identifiable Information (PII) Detection and Redaction**.

---

## Overview & Architecture

- **Base Model**: `Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2`
- **Backbone**: `MiniLM-L6-v2` (6 transformer layers, hidden dimension $H=384$, 12 attention heads)
- **Vocabulary**: 30,522 WordPiece tokens (only **0.71 MB** tokenizer footprint)
- **Target Deployment**: Mobile (Flutter, Android, iOS), WebAssembly, and Edge devices
- **Latency**: **~3–5 ms** per sentence on mobile CPU (4x–5x faster than DeBERTa/GLiNER)
- **Peak RAM Overhead**: **< 35 MB**

---

## Model Benchmarks & Comparison

| Model Asset | Precision | File Size | Latency (Mobile CPU) | Memory Overhead | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `minilm_l6_pii.slim.onnx` | FP32 | **90.54 MB** | ~8–12 ms | ~95 MB | Cleaned & constant-folded via `onnxslim` |
| `minilm_l6_pii.slim.int8.onnx` | INT8 | **~22 MB** | **~3–5 ms** | **< 35 MB** | **Recommended Mobile Default** (Dynamic `QUInt8`) |
| `tokenizer.json` | Tokenizer | **0.71 MB** | — | — | Ultra-compact WordPiece tokenizer |
| `vocab.txt` | Vocab | **0.23 MB** | — | — | Standard 30k vocabulary mapping |

---

## 55 Supported PII Entity Classes

The model detects 55 fine-grained entity types directly in a single forward pass:

### 1. Healthcare & Clinical (HIPAA Compliance)
- `medical_record_number` (MRN)
- `health_plan_beneficiary_number`
- `blood_type`
- `biometric_identifier`

### 2. Personal & Contact Information
- `first_name`, `last_name`, `user_name`
- `date_of_birth`, `age`, `date`, `time`, `date_time`
- `phone_number`, `fax_number`, `email`
- `street_address`, `city`, `state`, `county`, `country`, `postcode`, `coordinate`
- `gender`, `language`, `education_level`, `occupation`, `employment_status`

### 3. Government & Identity Documents
- `ssn` (Social Security Number)
- `national_id`, `tax_id`
- `certificate_license_number`, `license_plate`, `vehicle_identifier`

### 4. Financial & Digital Identifiers
- `credit_debit_card`, `account_number`, `bank_routing_number`, `swift_bic`, `cvv`, `pin`
- `ip_address` (`ipv4`, `ipv6`), `mac_address`, `device_identifier`
- `api_key`, `password`, `url`, `http_cookie`

---

## Usage Instructions

### 1. Python (ONNX Runtime) Quickstart

```python
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 1. Load Tokenizer & Model Config
tokenizer = AutoTokenizer.from_pretrained("./output_models")
with open("./output_models/config.json", "r") as f:
    config = json.load(f)
id2label = config["id2label"]

# 2. Initialize ONNX Runtime Session
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
session = ort.InferenceSession(
    "./output_models/minilm_l6_pii.slim.int8.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)

# 3. Tokenize Input Text
text = "Patient John Doe (DOB: 12/05/1982, MRN: 948201) was admitted to Seattle Grace Hospital. Ph: (555) 234-5678."
inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)

# 4. Run Inference
ort_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
}
logits = session.run(None, ort_inputs)[0]  # [1, seq_len, 111]

# 5. Decode Predicted Entity Spans
pred_ids = np.argmax(logits[0], axis=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print(f"Input: {text}\n")
print("Detected PII Spans:")
for token, pred_id in zip(tokens, pred_ids):
    label = id2label[str(pred_id)]
    if label != "O":
        print(f"  {token:<15} -> {label}")
```

---

### 2. Flutter Deployment (`flutter_onnxruntime`)

Add dependency to `pubspec.yaml`:
```yaml
dependencies:
  flutter_onnxruntime: ^1.9.0
```

Run inference on device:
```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

Future<void> detectPii(List<int> inputIds, List<int> attentionMask) async {
  final ort = OnnxRuntime();

  // Load model from assets or local storage
  final session = await ort.createSessionFromAsset(
    'assets/models/minilm_l6_pii.slim.int8.onnx',
  );

  final inputIdsTensor = await OrtValue.fromList(inputIds, [1, inputIds.length]);
  final maskTensor = await OrtValue.fromList(attentionMask, [1, attentionMask.length]);

  try {
    final outputs = await session.run({
      'input_ids': inputIdsTensor,
      'attention_mask': maskTensor,
    });

    final logitsTensor = outputs['logits'];
    if (logitsTensor != null) {
      final logitsList = await logitsTensor.asList();
      // Decode argmax labels with config.json id2label
    }

    for (final t in outputs.values) {
      t.dispose();
    }
  } finally {
    inputIdsTensor.dispose();
    maskTensor.dispose();
    await session.close();
  }
}
```

---

## Pipeline Reproduction

To reproduce the download, graph simplification, and INT8 quantization:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install huggingface_hub onnx onnxruntime onnxslim requests
uv run python optimize_models.py         # Skips existing assets
uv run python optimize_models.py --force # Forces re-download and re-optimization
```
