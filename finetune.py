import torch
from datasets import load_dataset, Audio, load_metric, DatasetDict
from typing import List, Dict, Union
from transformers import Wav2Vec2Processor, AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer, HubertForCTC
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass
import re
from src.modelutils import get_layers, get_model
from src.aq import QuantizedLinear, set_layer_precision
import numpy as np
import types
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import argparse
import time
import os


def make_mixed_precision(model, num_precisions=1):
    """Add architecture parameters to the quantized model"""
    _HIDDEN_STATES_START_POSITION = 1

    def org_forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def subforward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        if not self.training:
            return self.org_forward(
                input_values,
                attention_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
                labels,
            )

        logits_list = []
        for prec in range(self.num_precisions-1, -1, -1):
            set_layer_precision(self, prec)
            logits_list.append(self.subforward(
                input_values,
                attention_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
                labels,
            ))

        loss = None
        losses = []
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            for logits in logits_list:
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                losses.append(loss)

        if len(logits_list) > 1:
            kl_loss = self.calculate_kl_loss(logits_list)
            loss = losses[0] + sum(losses[1:]) * 0.1 + sum(kl_loss)
        else:
            loss = losses[0]

        return CausalLMOutput(
            loss=loss, logits=logits,
        )

    def calculate_kl_loss(self, logits):
        odim = logits[0].shape[-1]
        probs_largest = F.softmax(logits[0], dim=-1, dtype=torch.float32).detach()
        loss_kl = []
        for i, logit in enumerate(logits[1:]):
            log_probs_sub = F.log_softmax(logit, dim=-1, dtype=torch.float32)
            loss_kl.append(0.005 * F.kl_div(log_probs_sub.view(-1, odim), probs_largest.view(-1, odim), reduction='sum'))

        return loss_kl

    def _add_functions(layer):
        if isinstance(layer, HubertForCTC):
            layer.org_forward = types.MethodType(org_forward, layer)
            layer.subforward = types.MethodType(subforward, layer)
            layer.forward = types.MethodType(forward, layer)
            layer.calculate_kl_loss = types.MethodType(calculate_kl_loss, layer)
            layer.num_precisions = num_precisions

    model.apply(_add_functions)


chars_to_ignore_regex = '[\,\?\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]) + " "
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


parser = argparse.ArgumentParser(description='Fine-tune quantized HuBERT model for ASR task')

parser.add_argument('--num_precisions', type=int, default=2, help='Number of precisions for quantization')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path or HuggingFace name of the base HuBERT model')
parser.add_argument('--quant_model', type=str, default="",
                    help='Path to quantized model directory (absolute or relative; empty = use base model weights)')
parser.add_argument('--output_dir', type=str, default="./results/test", help='Directory to save the output')
parser.add_argument('--seed', type=int, default=1000, help='Random seed')
parser.add_argument('--dataloader_num_workers', type=int, default=2, help='Number of dataloader workers')
parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
parser.add_argument('--use_fp16', action='store_true', default=False, help='Whether to use fp16 (half-precision)')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to enable resume')
parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to apply gradient checkpointing")

args = parser.parse_args()


num_precisions = args.num_precisions
base_model = args.base_model
quant_model_path = args.quant_model
output_dir = args.output_dir
seed = args.seed
dataloader_num_workers = args.dataloader_num_workers
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
per_device_eval_batch_size = args.per_device_eval_batch_size
num_train_epochs = args.num_train_epochs
learning_rate = args.learning_rate
warmup_ratio = args.warmup_ratio
use_fp16 = args.use_fp16
resume = args.resume
gradient_checkpointing = args.gradient_checkpointing


cache_path = os.environ.get("LIBRISPEECH_CACHE_DIR", "./data_cache/")
data_path  = os.environ.get("LIBRISPEECH_DATA_DIR",  "/path/to/librispeech/")
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_scripts/')

ds_librispeech = load_dataset(script_path+'librispeech_asr_train.py', 'clean', data_dir=data_path, cache_dir=cache_path, trust_remote_code=True)
ds_test = load_dataset(script_path+'librispeech_asr_test.py', 'all', data_dir=data_path, cache_dir=cache_path, trust_remote_code=True)
librispeech = DatasetDict({"train": ds_librispeech["train.100"], "validation": ds_librispeech["validation"]})
testdata = DatasetDict({"dev_clean": ds_test["validation.clean"], "dev_other": ds_test["validation.other"], "test_other": ds_test["test.other"], "test_clean": ds_test["test.clean"]})

librispeech = librispeech.map(remove_special_characters)
testdata = testdata.map(remove_special_characters)

librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))
testdata = testdata.cast_column("audio", Audio(sampling_rate=16000))

processor = AutoProcessor.from_pretrained(base_model)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

librispeech = librispeech.map(prepare_dataset, remove_columns=librispeech.column_names["train"], num_proc=20)
testdata = testdata.map(prepare_dataset, remove_columns=testdata.column_names[list(testdata.keys())[0]], num_proc=20)

model = get_model(
        base_model,
        quant_model_path.strip() if quant_model_path.strip() else None,
        trust_remote_code=True,
    ).to("cuda")
make_mixed_precision(model, num_precisions)

training_args = TrainingArguments(
    output_dir=output_dir,
    seed=seed,
    dataloader_num_workers=dataloader_num_workers,
    group_by_length=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    fp16=use_fp16,
    gradient_checkpointing=gradient_checkpointing,
    save_steps=1,
    eval_steps=1,
    logging_steps=1_000,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    save_total_limit=2,
    push_to_hub=False,
    greater_is_better=False,
    length_column_name="input_length",
    save_strategy='epoch',
    resume_from_checkpoint=resume,
)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = pred_logits

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, (tuple, list)):
        preds_ids_list = []
        for logit in logits:
            preds_ids_list.append(torch.argmax(logit, dim=-1))
        return tuple(preds_ids_list)
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=librispeech["train"],
    eval_dataset=librispeech["validation"],
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print("Starting training...")
tick = time.time()
trainer.train()
tock = time.time()
print(f"Training completed in {tock - tick} seconds.")

model.eval()
subsets = ["dev_clean", "dev_other", "test_clean", "test_other"]
for prec in range(num_precisions - 1, -1, -1):
    set_layer_precision(model, prec)
    print(f"Evaluating with precision {prec}...")
    for subset in subsets:
        if subset in testdata:
            eval_result = trainer.evaluate(eval_dataset=testdata[subset])
            print(f"Evaluation WER results for {subset}: {eval_result}")
        else:
            print(f"Subset {subset} not found in the dataset.")
