import logging
import pickle

import evaluate

logging.basicConfig(level=logging.INFO)

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

import datasets
import numpy as np

import transformers
from sklearn.metrics import recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, PredictionOutput
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from codes.Models.MultitaskModel import MultitaskModel
from codes.Dataset.MLMDataset import MLMDataset
from codes.Dataset.RNMIDataset import RNMIDataset

from codes.utils.MultiTasks import MultitaskTrainer, NLPDataCollator
from codes.Dataset.RDNMIDataset import RDNMIDataset
from codes.utils.OtherUtils import add_special_tokens_to_tokenizer, get_dataset_dict, CodeChangeArguments, \
    create_path_if_not_exist
from codes.Config import cfg
from codes.Dataset.DDTPDataset import DDTPDataset


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def run_multitask_pretraining():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CodeChangeArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, code_change_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, code_change_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation training_args {training_args}")
    logger.info(f"Training/evaluation model_args {model_args}")
    logger.info(f"Training/evaluation code_change_args {code_change_args}")
    logger.info(f"Training/evaluation data_args {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    add_special_tokens_to_tokenizer(tokenizer)

    # Get the dataset
    dataset_dict = {
        "MLM": get_dataset_dict(code_change_args, tokenizer, MLMDataset),
        # "RNMI": get_dataset_dict(code_change_args, tokenizer, RDNMIDataset),
        "RNMI": get_dataset_dict(code_change_args, tokenizer, RNMIDataset),
        # "DDTP": get_dataset_dict(code_change_args, tokenizer, DDTPDataset),
    }

    multitask_model = MultitaskModel.create(
        model_name=model_args.model_name_or_path,
        model_type_dict={
            "MLM": transformers.AutoModelForMaskedLM,
            "RNMI": transformers.AutoModelForSequenceClassification,
            # "DDTP": transformers.AutoModelForMaskedLM,
        },
        model_config_dict={
            "MLM": transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
            "RNMI": transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=2),
            # "DDTP": transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
        },
    )
    for k in dataset_dict.keys():
        multitask_model.taskmodels_dict[k].resize_token_embeddings(len(tokenizer))

    # Preprocessing the raw_datasets
    sentence1_key = text_column_name = "text"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    padding = "max_length"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_MLM(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        tokenized_examples = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
        return tokenized_examples

    def preprocess_RNMI(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    def preprocess_DDTP(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        tokenized_examples = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
        tokenized_examples["DDTP"] = [1] * len(tokenized_examples["input_ids"])
        return tokenized_examples

    convert_func_dict = {
        "MLM": preprocess_MLM,
        "RNMI": preprocess_RNMI,
        # "DDTP": preprocess_DDTP,
    }

    columns_dict = {
        "MLM": ['input_ids', 'attention_mask', 'special_tokens_mask'],
        "RNMI": ['input_ids', 'attention_mask', 'labels'],
        # "DDTP": ['input_ids', 'attention_mask', 'special_tokens_mask', 'DDTP'],
    }
    tasks_proportion = {
        "MLM": 2,
        "RNMI": 1,
        # "DDTP": 1,
    }
    logger.info(f"Training/evaluation tasks_proportion {tasks_proportion}")

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            with training_args.main_process_first(desc="dataset map pre-processing"):
                features_dict[task_name][phase] = phase_dataset.map(
                    convert_func_dict[task_name],
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )

    if training_args.do_train:
        train_dataset = {
            task_name: dataset["train"]
            for task_name, dataset in features_dict.items()
        }
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = {
            task_name: dataset["validation"]
            for task_name, dataset in features_dict.items()
        }
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        predict_dataset = {
            task_name: dataset["test"]
            for task_name, dataset in features_dict.items()
        }
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for task_name, dataset in train_dataset.items():
            for index in random.sample(range(len(dataset)), 3):
                logger.info(f"Sample {index} of the training set in task {task_name}: {dataset[index]}.")

    def convert_eval_pred(p):
        if isinstance(p.predictions, list):
            logits = p.predictions
            logits = np.hstack(logits)
            # logits = np.concatenate(logits, axis=0)
            logits = logits.reshape(p.label_ids.shape)
            p.predictions = logits
        return p

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics_cls(p: Union[EvalPrediction, PredictionOutput]):
        p = convert_eval_pred(p)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y_preds = np.argmax(preds, axis=1)

        y_trues = list(map(lambda x: int(x), p.label_ids))

        recall = recall_score(y_trues, y_preds, average='binary')
        precision = precision_score(y_trues, y_preds, average='binary')
        f1 = f1_score(y_trues, y_preds, average='binary')
        eval_result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
        }

        logger.info("***** Eval results *****")
        for key in sorted(eval_result.keys()):
            logger.info("  %s = %s", key, str(round(eval_result[key], 4)))

        return eval_result

    metric = evaluate.load("accuracy")

    def compute_metrics_MLM(eval_preds):
        eval_preds = convert_eval_pred(eval_preds)
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    def compute_metrics_RNMI(p: Union[EvalPrediction, PredictionOutput]):
        p = convert_eval_pred(p)
        y_preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        y_trues = list(map(lambda x: int(x), p.label_ids))

        recall = recall_score(y_trues, y_preds, average='binary')
        precision = precision_score(y_trues, y_preds, average='binary')
        f1 = f1_score(y_trues, y_preds, average='binary')
        eval_result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
        }

        logger.info("***** Eval results *****")
        for key in sorted(eval_result.keys()):
            logger.info("  %s = %s", key, str(round(eval_result[key], 4)))

        return eval_result

    compute_metrics = {
        "MLM": compute_metrics_MLM,
        "RNMI": compute_metrics_RNMI,
        # "DDTP": compute_metrics_MLM,
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator(tokenizer=tokenizer, mlm_probability=0.15),
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tasks_proportion=tasks_proportion
    )

    # Training
    if training_args.do_train:
        # print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
        # print(multitask_model.taskmodels_dict["MLM"].roberta.embeddings.word_embeddings.weight.data_ptr())
        # print(multitask_model.taskmodels_dict["RNMI"].roberta.embeddings.word_embeddings.weight.data_ptr())
        # print(multitask_model.taskmodels_dict["DDTP"].roberta.embeddings.word_embeddings.weight.data_ptr())
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model_dir = os.path.join(training_args.output_dir, "multitask_model")
        create_path_if_not_exist(model_dir)

        with open(os.path.join(model_dir, "saved_model.pkl"), "wb") as dump_file:
            pickle.dump(multitask_model, dump_file)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]

        for eval_dataset, task in zip(predict_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(predict_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(predict_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return model_args, data_args, training_args, code_change_args, metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_multitask_pretraining()


if __name__ == "__main__":
    # run_multitask_pretraining()
    run_multitask_pretraining()
