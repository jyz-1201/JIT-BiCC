import argparse

import math
import numpy as np
from datasets import Dataset
from numpy import int64
from transformers import RobertaTokenizer

from codes.Dataset.JITDPDataset import JITDPDataset
from codes.utils.OtherUtils import get_dataset_dict

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--train_data_file", type=str, required=True,
                    help="The input training data file (a text file).")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--eval_data_file", type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
parser.add_argument("--test_data_file", type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

parser.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")

parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Run evaluation during training at each logging step.")

parser.add_argument("--train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--do_seed', type=int, default=123456,
                    help="random seed for data order initialization")
parser.add_argument('--epochs', type=int, default=1,
                    help="training epochs")

parser.add_argument('--feature_size', type=int, default=14,
                    help="Number of features")
parser.add_argument('--num_labels', type=int, default=2,
                    help="Number of labels")
parser.add_argument('--head_dropout_prob', type=float, default=0.1,
                    help="Number of labels")
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                    help="Number of labels")
parser.add_argument('--max_msg_length', type=int, default=64,
                    help="Number of labels")
parser.add_argument('--patience', type=int, default=5,
                    help='patience for early stop')

args = parser.parse_args()
args.no_abstraction = True

# add new special tokens
tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
tokenizer.add_special_tokens(special_tokens_dict)

msg_boundary_id = tokenizer('[ADD]').data["input_ids"][1]

datasets = get_dataset_dict(args, tokenizer, JITDPDataset)

statistic = []

for type_of_data in datasets:
    dataset = datasets[type_of_data]
    for example in dataset:
        text = example["text"]
        msg_boundary = text.find("[ADD]")
        if msg_boundary == -1:
            msg_boundary = text.find("[DEL]")
        msg = text[: msg_boundary]
        code = text[msg_boundary:]
        msg_tokens = msg.lower().split()
        code_tokens = code.lower().split()
        cnt = 0
        for msg_token in msg_tokens:
            if msg_token in code_tokens:
                cnt += 1
        statistic.append(math.ceil(cnt / len(msg_tokens) * 100))
        # statistic.append(cnt)

counter = np.zeros(max(statistic)+10, int64)
for e in statistic:
    counter[e] += 1
print(counter)


