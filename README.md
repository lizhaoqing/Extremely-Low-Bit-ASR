# CSVQ: Codebook-based Speech Vector Quantization

Multi-precision additive quantization for HuBERT speech models, evaluated on LibriSpeech CTC.  
This codebase is built upon [AQLM](https://github.com/Vahe1994/AQLM).

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

Set the following environment variables before running:

```bash
export LIBRISPEECH_DATA_DIR=/path/to/librispeech/   # directory containing LibriSpeech/train-clean-100/, etc.
export LIBRISPEECH_CACHE_DIR=./data_cache/           # HuggingFace datasets cache (created automatically)
```

The `data_scripts/` folder contains the HuggingFace dataset scripts that read LibriSpeech from a local directory.

## Usage

### Step 1: Quantize HuBERT

Edit `run_quantize.sh` to set `MODEL_PATH` (path to `hubert-large-960h`), then:

```bash
bash run_quantize.sh
```

Key arguments for `quantize.py`:

| Argument | Description |
|---|---|
| `--num_codebooks` | Comma-separated list of codebooks per precision level, e.g. `"2,1,1"` for 2/3/4-bit |
| `--nbits_per_codebook` | Bits per codebook entry, e.g. `"8,8,8"` |
| `--num_precisions` | Number of precision levels (length of the above lists) |
| `--nsamples` | Number of calibration samples |
| `--in_group_size` | Input feature group size for quantization |
| `--save` | Directory to save quantized layers |
| `--resume` | Resume from previously saved layers |

### Step 2: Finetune quantized HuBERT (optional)

```bash
bash run_finetune.sh
```

Key arguments for `finetune.py`:

| Argument | Description |
|---|---|
| `--base_model` | Path to the original HuBERT model |
| `--quant_model` | Path to the quantized model directory (output of Step 1) |
| `--num_precisions` | Must match the value used in Step 1 |

## Acknowledgement

This codebase is built upon [AQLM](https://github.com/Vahe1994/AQLM).
