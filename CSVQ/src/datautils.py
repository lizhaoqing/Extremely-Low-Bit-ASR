import os
import random
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import AutoProcessor
import re

# Data paths — override via environment variables if needed:
#   LIBRISPEECH_DATA_DIR  : root directory of the LibriSpeech corpus
#   LIBRISPEECH_CACHE_DIR : HuggingFace datasets cache directory
_LIBRISPEECH_DATA_DIR  = os.environ.get("LIBRISPEECH_DATA_DIR",  "/path/to/librispeech/")
_LIBRISPEECH_CACHE_DIR = os.environ.get("LIBRISPEECH_CACHE_DIR", "./data_cache/")
_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_scripts/')


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


chars_to_ignore_regex = '[\,\?\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]) + " "
    return batch


def get_librispeech_train(nsamples, seqlen, processor, eval_mode=False, model=None):
    assert not eval_mode, "Only train set is supported in librispeech"

    traindata = load_dataset(
        _SCRIPT_DIR + 'librispeech_asr_train.py',
        'clean',
        data_dir=_LIBRISPEECH_DATA_DIR,
        split="train.100",
        cache_dir=_LIBRISPEECH_CACHE_DIR,
        trust_remote_code=True,
    )
    traindata = traindata.map(remove_special_characters)
    traindata = traindata.cast_column("file", Audio(sampling_rate=16000))
    trainloader = []

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            if len(traindata[i]['audio']["array"]) >= seqlen:
                trainenc = processor(
                    traindata[i]['audio']["array"],
                    sampling_rate=traindata[i]['audio']["sampling_rate"],
                    return_tensors="pt",
                )
                break
        i = random.randint(0, trainenc.input_values.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_values[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader


def _make_datasets_for_test(task_data, processor, return_att_mask=False):

    def prepare_dataset(batch):
        audio = batch["file"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        if return_att_mask:
            batch["attention_mask"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    processed = task_data
    processed = processed.map(remove_special_characters)
    processed = processed.cast_column("file", Audio(sampling_rate=16000))
    processed = processed.map(prepare_dataset, remove_columns=processed.column_names[list(processed.keys())[0]], num_proc=20)

    return processed


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=512,
    eval_mode=False,
    model_path=None,
    trust_remote_code=None,
    model=None,
):

    set_seed(seed)
    processor = AutoProcessor.from_pretrained(model_path)

    seqlen = int((seqlen+1) * 320)

    if not eval_mode:
        data = get_librispeech_train(nsamples, seqlen, processor, eval_mode, model)
        print(f"Loaded data from {name}; {len(data)} sequences")
    else:
        ds = load_dataset(_SCRIPT_DIR + 'librispeech_asr_test.py', 'all',
                          data_dir=_LIBRISPEECH_DATA_DIR, cache_dir=_LIBRISPEECH_CACHE_DIR,
                          trust_remote_code=True)
        test_data = DatasetDict({
            "dev_clean":  ds["validation.clean"],
            "dev_other":  ds["validation.other"],
            "test_other": ds["test.other"],
            "test_clean": ds["test.clean"],
        })
        data = _make_datasets_for_test(test_data, processor, return_att_mask=False)
        total = sum(len(data[s]) for s in data)
        print(f"Loaded eval data from {name}; {total} sequences across {list(data.keys())}")
    return data, processor
