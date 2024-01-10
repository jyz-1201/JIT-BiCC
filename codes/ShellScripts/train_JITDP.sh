date=0202
script_output_dir="output/models/model_JITDP_on_MLM_${date}"

python JITFine/semantic/run.py \
--output_dir=${script_output_dir} \
--config_name=microsoft/codebert-base \
--model_name_or_path=output/models/model_MLM_0130 \
--tokenizer_name=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl \
--eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl\
--test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl\
--epoch 50 \
--max_seq_length 512 \
--max_msg_length 64 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--patience 10 \
--seed 42 2>&1| tee "${script_output_dir}/train.log"
