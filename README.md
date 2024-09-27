# JIT-BiCC-replication-package



This repository contains source code that we used to perform experiment in paper titled "Just-In-Time Software Defect Prediction via Bi-modal Change Representation Learning".


Please follow the steps below to reproduce the result.

## Environment Setup

### Environment Requirement

[Python](https://www.python.org/downloads/) and [Anaconda](https://docs.anaconda.com/anaconda/install/) are required to setup the environment. The required python packages are stored in ```./requirements.yml```, which can be aotumatcically installed by anaconda. We recommand to setup the environment on a Linux machine.

### Python Environment Setup

Run the following command in terminal (or command line) to prepare virtual environment. The required version of python and packages will be installed and activated.

```shell
conda env create --file requirements.yml
conda activate jitbicc
```

## Experiment Result Replication Guide

### **RQ1 Implementation**

There are 9 comparative approaches in RQ1. To reproduce the results of baselines, run the following commands:

- LApredict

  ```shell
  python -m baselines.LApredict.lapredict
  ```

- Deeper

  ```shell
  python -m baselines.Deeper.deeper
  ```

- DeepJIT

  ```shell
  python -m baselines.DeepJIT.deepjit
  ```

- CC2Vec

  ```shell
  python -m baselines.CC2Vec.cc2vec
  ```

- JITLine

  ```shell
  python -m baselines.JITLine.jitline_rq2 -style concat
  ```
  
- JITFine
  
  To train JIT-Fine:
  
  ```shell
  python -m JITFine.concat.run \
      --output_dir=model/jitfine/saved_models_concat/checkpoints \
      --config_name=microsoft/codebert-base \
      --model_name_or_path=microsoft/codebert-base \
      --tokenizer_name=microsoft/codebert-base \
      --do_train \
      --train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl \
      --eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl\
      --test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl\
      --epoch 50 \
      --max_seq_length 512 \
      --max_msg_length 64 \
      --train_batch_size 24 \
      --eval_batch_size 128 \
      --learning_rate 1e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --feature_size 14 \
      --patience 10 \
      --seed 42 2>&1| tee model/jitfine/saved_models_concat/train.log
  
  ```
  
  To obtain the evaluation:
  
  ```shell
  python -m JITFine.concat.run \
      --output_dir=model/jitfine/saved_models_concat/checkpoints \
      --config_name=microsoft/codebert-base \
      --model_name_or_path=microsoft/codebert-base \
      --tokenizer_name=microsoft/codebert-base \
      --do_test \
      --train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl \
      --eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl\
      --test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl\
      --epoch 50 \
      --max_seq_length 512 \
      --max_msg_length 64 \
      --train_batch_size 256 \
      --eval_batch_size 25 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --only_adds \
      --buggy_line_filepath=data/jitfine/changes_complete_buggy_line_level.pkl \
      --seed 42 2>&1 | tee model/jitfine/saved_models_concat/test.log
  
  ```

- CodeReviewer

  ```shell
  bash ./codes/ShellScripts/train_JITDP_with_t5.sh
  ```

- JIT-BiCC

  ```shell
  bash ./codes/ShellScripts/train_multitask.sh
  ```


### **RQ2 Implementation**

- To run JIT-BiCC without both RMI and MLM, all the commands (and the results) are exactly the same as running baseline JIT-Fine.

- To run JIT-BiCC without RMI, execute the following commands sequentially:

  ```shell
  bash ./codes/ShellScripts/train_MLM.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_JITDP_with_expert_v2.sh
  ```
  where the argument "model_name_or_path" in scripts "./codes/ShellScripts/train_JITDP_with_expert_v2.sh" should be changed to your actual model file path.

- To run JIT-BiCC without MLM, execute the following commands sequentially:

  ```shell
  bash ./codes/ShellScripts/train_RDNMI_on_CodeBERT.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_JITDP_with_expert_v2.sh
  ```
  where the argument "model_name_or_path" in scripts "./codes/ShellScripts/train_JITDP_with_expert_v2.sh" should be changed to your actual model file path.

### **RQ3 Implementation**

- Training Order of Pretraining Objectives:

  To experiment on RMI -> MLM, execute the following commands sequentially:
  
  ```shell
  bash ./codes/ShellScripts/train_RDNMI_on_CodeBERT.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_MLM_on_RDNMI.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_JITDP_with_expert_v2.sh
  ```
  
  where arguments related to the model file path in all bash scripts that you run should be changed to your actual path, and argument "num_train_epochs" in script "./codes/ShellScripts/train_RDNMI_on_CodeBERT.sh" should be set to 100.

  To experiment on MLM -> RMI, execute the following commands sequentially:
  
  ```shell
  bash ./codes/ShellScripts/train_MLM.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_RDNMI.sh
  ```
  
  ```shell
  bash ./codes/ShellScripts/train_JITDP_with_expert_v2.sh
  ```
  where arguments related to the model file path in all bash scripts that you run should be changed to your actual path.


- Objective Proportion:
  In file "./codes/PythonScripts/run_multitask.py", change the value of variable "tasks_proportion" (in method run_multitask_pretraining) 
  to "1:1", "2:1", "3:1" separately, and rerun the command below each time that you change this variable:
  
  ```shell
  bash ./codes/ShellScripts/train_multitask.sh
  ```
