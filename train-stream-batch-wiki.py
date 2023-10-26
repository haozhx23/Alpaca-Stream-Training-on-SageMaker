#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

############################
from datetime import datetime
import os
import deepspeed
import json, socket
import argparse
from datasets import load_dataset
import datasets
from functools import partial
from typing import Dict, Optional, Sequence, Any
import time
############################


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",   ########### batch
            # padding='max_length',  ########### stream
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




############################
def get_local_files_list(file_path):
    import pathlib
    localnvme = pathlib.Path(file_path)
    fileslist = list(set([str(i) if not i.is_dir() else '#' for i in localnvme.rglob("*")]))
    if '#' in fileslist:
        fileslist.remove('#')
    # print('-NVMEfileslist-', fileslist)
    return fileslist
############################


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.Tensor([torch.nn.utils.rnn.pad_sequence(
            input_id, batch_first=True, padding_value=self.tokenizer.pad_token_id) for input_id in input_ids])
        labels = [torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=IGNORE_INDEX) for label in labels]
        attention_mask = [input_id.ne(self.tokenizer.pad_token_id) for input_id in input_ids]

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids[0], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels[0], batch_first=True, padding_value=IGNORE_INDEX)
        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def tokenize_wiki_batch(
        tokenizer: transformers.PreTrainedTokenizer,
        list_data_dict: Dict[str, Any],
):

    txtinput = "\n{title}\n\n\t\t{text}\n\n"

    # example = list_data_dict
    # print('---DDD---',list_data_dict['text'])
    
    titlelist = list_data_dict['title']
    textlist = list_data_dict['text']
    batchlen = len(list_data_dict['text'])

    sources = [
        f"{tokenizer.bos_token}" for _ in titlelist
    ]
    # targets = [txtinput.format_map(example) if example.get("title", "") != "" else example.get("text", "")
    #           for example in list_data_dict
    #           ]
    
    
    targets = [txtinput.format_map({'title':titlelist[i],'text':textlist[i]}) if textlist[i] != "" else "\n"
              for i in range(batchlen)
              ]
        

    # logging.warning("Tokenizing inputs... This may take some time...")
    # print('---DDD---',len(sources),len(targets))
    data_dict = preprocess(sources, targets, tokenizer)
    
    input_ids = data_dict["input_ids"]
    labels = data_dict["labels"]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #############Copy model artifacts from S3 to NVME###############
    if 0 == LOCAL_RANK:
        os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH'], model_args.model_name_or_path))
        print(f'------rank {LOCAL_RANK} finished cp-------')

    torch.distributed.barrier()
    ############################

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        
        legacy=False ############ p5 test
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

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    #############Load Dataset in Stream###############
    tot_num_samples = int(os.environ['TOTAL_NUM_SAMPLES'])
    train_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * WORLD_SIZE
    tot_steps = int(tot_num_samples / train_batch_size)  # floor as drop_last
    training_args.max_steps = tot_steps

    fileslist = {'train': get_local_files_list(data_args.data_path)}
    train_data = load_dataset('json', data_files=fileslist, split='train', streaming=True)

    train_data = train_data.shuffle(buffer_size=1000000).map(
        partial(
            # tokenize_prompt_gen,
            tokenize_wiki_batch,
            tokenizer
        ),
        batched=True
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_module = dict(train_dataset=train_data, eval_dataset=None, data_collator=data_collator)
    data_module = dict(train_dataset=train_data.with_format("torch"), eval_dataset=None, data_collator=data_collator)
    
    ############################
    print('-------DD-------1: ', time.time())
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    print('-------DD-------3: ', time.time())
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)


    ##############Save model at node0-rank0 ##############
    if WORLD_RANK == 0:
        persistant_path = os.environ['OUTPUT_MODEL_S3_PATH'] + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")) + '/'
        os.system("./s5cmd sync {0} {1}".format(training_args.output_dir, persistant_path))

    torch.distributed.barrier()
    ############################


if __name__ == "__main__":
    # Environment variables set by torchrun OR torch.distributed.launch
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    # torch.cuda.set_device(LOCAL_RANK)
    deepspeed.init_distributed(dist_backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    # torch.distributed.init_process_group(backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    # print('os.environ["SM_CHANNEL_TRAIN1"]: ',os.environ["SM_CHANNEL_TRAIN1"])

    train()
