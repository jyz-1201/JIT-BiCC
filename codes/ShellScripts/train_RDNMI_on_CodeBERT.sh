date="0502"
PLM_name="CodeBERT"
script_output_dir="output/models/model_RDNMI_on_${PLM_name}_${date}"
JITDP_model_dir="output/models/model_JITDP_with_expert_on_RDNMI_on_${PLM_name}_${date}"

python codes/PythonScripts/run_RNMI.py \
--output_dir=${script_output_dir} \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_eval \
--do_predict \
--tokenizer_name=microsoft/codebert-base \
--model_type=RobertaSequenceClassification \
--overwrite_output_dir \
--logging_steps=100 \
--save_steps=500 \
--train_data_file="data/jitfine/changes_train.pkl&data/jitfine/features_train.pkl" \
--eval_data_file="data/jitfine/changes_valid.pkl&data/jitfine/features_valid.pkl" \
--test_data_file="data/jitfine/changes_test.pkl&data/jitfine/features_test.pkl" \
--max_seq_length 512 \
--max_msg_length 64 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=32 \
--num_train_epochs=300 \
--warmup_steps=500 \
--pad_to_max_length=False \
--metric_for_best_model=f1 \
--learning_rate 5e-4 \
--RNMI_noise_ratio 0.0 \
--overwrite_output_dir \
--evaluation_strategy=steps \
--save_strategy=steps \
--fp16 \
--data_seed=123456 \
--save_total_limit=2 \
--seed 42

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
