date="0214"
script_output_dir="output/models/model_JITDP_with_expert_on_MLM_${date}"

python codes/PythonScripts/run_JITDP.py \
--output_dir=${script_output_dir} \
--model_name_or_path=output/models/model_MLM_0130 \
--do_train \
--do_eval \
--do_predict \
--logging_steps=20 \
--train_data_file="data/jitfine/changes_train.pkl&data/jitfine/features_train.pkl" \
--eval_data_file="data/jitfine/changes_valid.pkl&data/jitfine/features_valid.pkl" \
--test_data_file="data/jitfine/changes_test.pkl&data/jitfine/features_test.pkl" \
--max_seq_length 512 \
--max_msg_length 64 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=1 \
--num_train_epochs=50 \
--warmup_steps=0 \
--pad_to_max_length=False \
--model_type=JITDPSemantic \
--load_best_model_at_end=True \
--metric_for_best_model=f1 \
--learning_rate 2e-5 \
--overwrite_output_dir \
--evaluation_strategy=epoch \
--save_strategy=epoch \
--fp16 \
--data_seed=123456 \
--seed 42 2>&1| tee "${script_output_dir}/train.log"

