date=0512
PLM_name="RDNMI_on_CodeBERT_0430_2"
script_output_dir="output/models/model_MLM_on_${PLM_name}_${date}"
JITDP_model_dir="output/models/model_JITDP_with_expert_on_MLM_on_${PLM_name}_${date}"

python codes/PythonScripts/run_MLM.py \
--output_dir=${script_output_dir} \
--do_eval \
--do_train \
--model_name_or_path="output/models/model_${PLM_name}" \
--overwrite_output_dir \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=32 \
--logging_steps=100 \
--save_steps=500 \
--learning_rate=5e-4 \
--evaluation_strategy=steps \
--save_strategy=steps \
--train_data_file="data/jitfine/changes_train.pkl&data/jitfine/features_train.pkl" \
--eval_data_file="data/jitfine/changes_valid.pkl&data/jitfine/features_valid.pkl" \
--test_data_file="data/jitfine/changes_test.pkl&data/jitfine/features_test.pkl" \
--seed=42 \
--data_seed=123456 \
--num_train_epochs=200 \
--warmup_steps=500 \
--line_by_line \
--save_total_limit=2 \
--fp16

python JITFine/concat/run.py \
--output_dir=${JITDP_model_dir} \
--config_name=microsoft/codebert-base \
--model_name_or_path=${script_output_dir} \
--tokenizer_name=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl \
--eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl \
--test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl \
--epoch 50 \
--max_seq_length 512 \
--max_msg_length 64 \
--train_batch_size 12 \
--eval_batch_size 32 \
--learning_rate 1e-5 \
--feature_size 14 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--patience 15 \
--seed 42 \
--only_adds \
--buggy_line_filepath=data/jitfine/changes_complete_buggy_line_level.pkl
