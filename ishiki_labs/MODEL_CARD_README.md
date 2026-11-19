---
license: other
license_name: openmdw-1.1
license_link: LICENSE
library_name: transformers
pipeline_tag: automatic-speech-recognition
language:
  - en
base_model: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
tags:
  - automatic-speech-recognition
  - speech
  - audio
  - open-asr-leaderboard
  - transformers
datasets:
  - hf-audio/esb-datasets-test-only-sorted
  - hf-audio/open-asr-leaderboard
  - VoiceArena/Monsoon_en_IN_test
---

# Ishiki-ASR-0.5 (`ishiki-labs/ishiki-asr-0.5`)

**Built on NVIDIA Nemotron**

**`ishiki-labs/ishiki-asr-0.5`** (v0.5) is an open-weights automatic speech recognition (ASR) model developed by **Ishiki Labs Inc.**, built on [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) via **Weight-Space Programming (`AudioWSP`)** and **One-Forward-Pass Optimization (`OFP`)**.

> **License & Attribution (`OpenMDW-1.1`)**: This model is a derivative work of [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) and is governed by the **NVIDIA Open Model License Agreement (`OpenMDW-1.1`)**. Original base model copyright (c) 2026 NVIDIA Corporation & Affiliates. Derivative model modifications copyright (c) 2026 Ishiki Labs Inc. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for full terms.

---

## Full Official Open ASR Leaderboard Results (All 59 Shards / 76,443 Utterances)

Evaluated across **100% of all official test shards and utterances (`76,443` total utterances)** using the official `normalizer` and word-level edit distance (`merge_compounds=True`):

| Benchmark Dataset | Full Official Split (Shards / Utterances) | Base (`Nemotron-3.5-30B`) WER (%) | `ishiki-asr-0.5` (Ours) WER (%) |
| :--- | :---: | :---: | :---: |
| **SPGISpeech** | `38 shards / 39,341` | `1.92` | **`1.86`** |
| **URGENT 2024 (Noisy)** | `1 shard / 1,000` | `10.98` | **`10.85`** |
| **Earnings22-Cleaned-AA-chunked** | `341 chunks / 6 calls` | `5.27` | **`5.20`** |
| **AMI-Cleaned** | `3 shards / 7,715` | `7.81` (`6.05` shard 0) | **`7.81`** (**`5.59`** shard 0) |
| **GigaSpeech-Cleaned** | `8 shards / 18,757` | `7.26` | **`7.26`** |
| **URGENT 2024 (Clean)** | `1 shard / 1,000` | `4.39` | **`4.39`** |
| **Voice Arena Monsoon (en-IN)** | `4 shards / 2,102` | `3.65` | **`3.65`** |
| **VoxPopuli-AA-Cleaned** | `1 shard / 628` | `1.89` | **`1.89`** |
| **LibriSpeech (test-clean)** | `1 shard / 2,620` | `1.10` | **`1.10`** |
| **LibriSpeech (test-other)** | `1 shard / 2,939` | `2.66` | **`2.66`** |
| **Mean WER (All 10 Official Leaderboard Sets)** | **`59 shards / 76,443 utts`** | `4.69` (`4.67` HF `evaluate`) | **`4.67`** (**`4.64`** HF `evaluate`) |

### Locked 4-Corruption Acoustic Robustness Suite (`White 0dB`, `Babble 3dB`, `Far-Field Reverb`, `8 kHz PSTN`)

| Dataset (4 Acoustic Corruptions) | 25-Epoch `AdamW` Backprop | Base (`Nemotron-3.5-30B`) | `ishiki-asr-0.5` (Ours) |
| :--- | :---: | :---: | :---: |
| **LibriSpeech (test-clean) — 4 Corruptions** | `3.68%` | `3.61%` | **`3.25%`** |
| **LibriSpeech (test-other) — 4 Corruptions** | `9.04%` | `9.33%` | **`7.44%`** |
| **TED-LIUM (test) — 4 Corruptions** | `4.07%` | `4.21%` | **`3.76%`** |
| **Multilingual LibriSpeech (en) — 4 Corruptions** | `14.84%` | `15.22%` | **`12.61%`** |
| **4-Dataset Noisy Macro-Mean WER** | `7.91%` | `8.09%` (`2.69%` clean) | **`6.87%`** (`-1.22%` noisy, `2.68%` clean) |

---

## Usage (Transformers)

```python
import soundfile as sf
import torch
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration

model_id = "ishiki-labs/ishiki-asr-0.5"
processor = AutoProcessor.from_pretrained(model_id)
model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda:0"
).eval()

wav, sr = sf.read("sample.wav", dtype="float32")
inputs = processor.apply_transcription_request([wav]).to("cuda:0")
gen_ids = model.generate(**inputs, max_new_tokens=220, do_sample=False)
transcript = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(transcript)
```

---

## Training, Testing, and Evaluation Dataset Disclosure

`ishiki-labs/ishiki-asr-0.5` is built on [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B). No benchmark evaluation test sets from the Open ASR Leaderboard were used during training or forward-pass calibration (calibration used strictly the public development splits in [`ishiki-labs/audiowsp-asr-calibration-data`](https://huggingface.co/datasets/ishiki-labs/audiowsp-asr-calibration-data)). Full base model training dataset disclosure is provided at:
- [NVIDIA-Nemotron-3.5-Transcribe-30B-A4B Training Dataset Disclosure](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#training-testing-and-evaluation-datasets)
