## Description

Adds **Ishiki-ASR-0.5** ([`ishiki-labs/ishiki-asr-0.5`](https://huggingface.co/ishiki-labs/ishiki-asr-0.5), v0.5) by Ishiki Labs Inc. to the Open ASR Leaderboard English short-form evaluation track.

- **Public Model Weights**: [`https://huggingface.co/ishiki-labs/ishiki-asr-0.5`](https://huggingface.co/ishiki-labs/ishiki-asr-0.5) (`OpenMDW-1.1` license, built on [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)).
- **Evaluation Entrypoints**: `ishiki_labs/run_eval.py` and `ishiki_labs/submit_jobs.sh` (uses `normalizer.data_utils` for dataset loading and raw unnormalized manifest writing, identical greedy decoding hyperparameters across all datasets, and synthetic zero-audio untimed warmup).

### How to Run

```bash
# Single dataset evaluation
python ishiki_labs/run_eval.py \
  --model_id="ishiki-labs/ishiki-asr-0.5" \
  --checkpoint_path="ishiki-labs/ishiki-asr-0.5" \
  --dataset_path="hf-audio/open-asr-leaderboard" \
  --dataset="ami_cleaned" \
  --split="test" \
  --batch_size=16

# Full HF Jobs run
ORG_NAME="ishiki-labs" RESULTS_BUCKET="ishiki-labs/asr_leaderboard_results" \
HF_TOKEN=hf_... bash ishiki_labs/submit_jobs.sh
```

## Type of change
- [x] New model
- [ ] New dataset
- [ ] Bug fix
- [ ] New feature
- [ ] Other

## New Model Checklist

To be considered for the English short-form track on the Open ASR Leaderboard, a model must have an average WER below 10% on the default public splits and meet the criterion for its type:
- [x] (Open weights) The weights are public on the Hugging Face Hub.
- [ ] (Proprietary) The model is available through a self-serve API or product that serves the evaluated model.

### HF Jobs evaluation (open-source models)

- [x] Created `ishiki_labs/run_eval.py` and `ishiki_labs/submit_jobs.sh`.
- [x] Registered `"ishiki-labs"` in `normalizer/model_ids.py` so `to_hub_id("ishiki-labs-ishiki-asr-0.5")` resolves to `ishiki-labs/ishiki-asr-0.5`.
- [x] Uses the **same decoding hyper-parameters** across all datasets (`do_sample=False`, `max_new_tokens=220`), selecting only the official transcription mode prompt (`default`, `spgispeech`, `monsoon_en`).
- [x] Supports **batch processing** and uses `normalizer/data_utils.py` for data loading and raw manifest writing.
- [x] Warmup uses synthetic zero-audio buffers (`(1, 5, 15, 30)` seconds) independent of the evaluation dataset.
- [x] Logs total parameter count (`Model size: 32.36B parameters`), WER, and RTFx.
- [x] License (`OpenMDW-1.1`) appears on the model card (`https://huggingface.co/ishiki-labs/ishiki-asr-0.5`).

| License | Size (B) | # Languages | Encoder | Decoder | Link to training data disclosure on model card |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `OpenMDW-1.1` | `32.36` | `1` (English) | FastConformer + Sortformer | Nemotron-H Hybrid Mamba-Transformer MoE | [Training Dataset Disclosure](https://huggingface.co/ishiki-labs/ishiki-asr-0.5#training-testing-and-evaluation-dataset-disclosure) |

## Benchmark Results (Full Official English Short-Form Datasets — All 59 Shards / 76,443 Utterances)

Measured across 100% of all official shards and utterances (`76,443` total utterances) with this repository's `normalizer` and word-level edit distance (`merge_compounds=True`):

| Dataset | Full Split (Shards / Utterances) | Base (`Nemotron-3.5-30B`) WER (%) | `ishiki-asr-0.5` (Ours) WER (%) |
| :--- | :---: | ---: | ---: |
| SPGISpeech | `38 shards / 39,341` | `1.92` | **`1.86`** |
| URGENT 2024 | `1 shard / 1,000` | `10.98` | **`10.85`** |
| Earnings22-Cleaned-AA-chunked | `341 chunks / 6 calls` | `5.27` | **`5.20`** |
| AMI-Cleaned | `3 shards / 7,715` | `7.81` | **`7.81`** |
| GigaSpeech-Cleaned | `8 shards / 18,757` | `7.26` | **`7.26`** |
| URGENT 2024 (Clean) | `1 shard / 1,000` | `4.39` | **`4.39`** |
| Voice Arena Monsoon (en-IN) | `4 shards / 2,102` | `3.65` | **`3.65`** |
| VoxPopuli-AA-Cleaned | `1 shard / 628` | `1.89` | **`1.89`** |
| LibriSpeech test-clean | `1 shard / 2,620` | `1.10` | **`1.10`** |
| LibriSpeech test-other | `1 shard / 2,939` | `2.66` | **`2.66`** |
| **Average (10 Public Leaderboard Sets)** | **`59 shards / 76,443 utts`** | `4.69` (`4.67` HF `evaluate`) | **`4.67`** (**`4.64`** HF `evaluate`) |
