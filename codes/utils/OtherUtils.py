import os
import random

import datasets
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataset_dict(args, tokenizer: AutoTokenizer, dataset_type):
    """
    convert datasets into a DatasetDict containing train/test/eval set
    """

    def get_dataset(mode):
        if mode == "train":
            file_path = args.train_data_file
        elif mode == "validation":
            file_path = args.eval_data_file
        else:
            file_path = args.test_data_file
        return Dataset.from_list(dataset_type(tokenizer, args, file_path=file_path, mode=mode))

    dataset_dict = datasets.DatasetDict({'train': get_dataset('train'),
                                         'validation': get_dataset('validation'),
                                         'test': get_dataset('test')})
    return dataset_dict


@dataclass
class CodeChangeArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    train_data_file: str = field(default=None, metadata={"help": "The input training data file."})
    eval_data_file: str = field(default=None, metadata={"help": "The input evaluation data file."})
    test_data_file: str = field(default=None, metadata={"help": "The input test data file."})
    cache_dataset_dir: str = field(
        default="cached_dataset",
        metadata={"help": "The directory for caching pre-processed datasets."})
    overwrite_cached_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite cached pre-processed dataset or not."},
    )
    no_abstraction: bool = field(
        default=True,
        metadata={"help": "Whether to do abstraction for the code change during data preprocess or not."},
    )
    max_msg_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum total input commit message length after tokenization. Messages longer "
                "than this will be truncated."
            )
        },
    )
    RNMI_masking_ratio: float = field(
        default=0.5,
        metadata={
            "help": (
                "The ratio of masked words in commit message for RNMI task."
            )
        },
    )
    RNMI_noise_ratio: float = field(
        default=0.5,
        metadata={
            "help": (
                "The ratio of masked words in commit message for RNMI task."
            )
        },
    )


def add_special_tokens_to_tokenizer(tokenizer):
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)


def return_true_with_probability(prob):
    rand_num = random.randint(0, 999)
    return rand_num <= prob * 1000



