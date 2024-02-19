import copy
import logging
from dataclasses import (
    dataclass, 
    field
)
from typing import (
    Dict, 
    Optional, 
    Sequence
)
from datasets import load_dataset
from config import LORA_COMPONENTS_MAP
from peft import (
    LoraConfig, 
    prepare_model_for_int8_training, 
    get_peft_model
)

import os
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    BitsAndBytesConfig
)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class LoggingArguments:
    project_name: str = field(
        default=None, metadata={"help": "The project name under which the experiment will be logged."}
    )
    wandb_token: str = field(
        default=None, metadata={"help": "API token for WandB hub."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum sequence length to be processed during instruction tuning."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters"
                "when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    llm_int8_threshold: float = field(
        default=6.0, metadata={"help": "The thresholf for a parameter to be designated a quantization outlier."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "The interpolation importance factor for the LoRA adapter."}
    )
    lora_r: int = field(
        default=8, metadata={"help": "The LoRA adapter rank."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout value for LoRA layers."}
    )
    def __post_init__(self):
        if self.model_name_or_path not in LORA_COMPONENTS_MAP.keys():
            raise ValueError(
                f"model_name_or_path argument must be one of the following: {LORA_COMPONENTS_MAP.keys()}"
            )


@dataclass
class DataArguments:
    hf_data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, hf_data_path: str, tokenizer: transformers.PreTrainedTokenizer, token: str, split="train"):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        dataset = load_dataset(hf_data_path, split=split, token=token)

        logging.warning("Formatting inputs...")
        sources = [f"### Instruction:\n{example['instruction']}\n\n### Response:\n" for example in dataset]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in dataset]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, hf_data_path=data_args.hf_data_path, token=model_args.token ,split="train")
    val_dataset = SupervisedDataset(tokenizer=tokenizer, hf_data_path=data_args.hf_data_path, token=model_args.token, split="validation")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoggingArguments))
    model_args, data_args, training_args, log_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        model_max_length=model_args.model_max_length,
        padding_side="right", 
        truncation_side="right"        
    )
    os.environ["WANDB_PROJECT"] = log_args.project_name
    os.environ["WANDB_API_KEY"] = log_args.wandb_token

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=model_args.llm_int8_threshold
    )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        quantization_config=quant_config,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    embedding_size = model_base.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model_base.resize_token_embeddings(len(tokenizer))

    embedding_size = model_base.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        input_embeddings = model_base.get_input_embeddings().weight.data
        output_embeddings = model_base.get_output_embeddings().weight.data
        model_base.resize_token_embeddings(len(tokenizer))
        input_embeddings_avg = input_embeddings[:embedding_size].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:embedding_size].mean(dim=0, keepdim=True)
        input_embeddings[embedding_size:] = input_embeddings_avg
        output_embeddings[embedding_size:] = output_embeddings_avg
    elif len(tokenizer) < embedding_size:
        model_base.resize_token_embeddings(len(tokenizer))

    model_base = prepare_model_for_int8_training(model_base)
    adapter_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_COMPONENTS_MAP[model_args.model_name_or_path]
    )
    model = get_peft_model(model_base, adapter_config)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
