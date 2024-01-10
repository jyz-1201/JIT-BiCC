date=0218

python codes/PythonScripts/run_MLM.py \
--output_dir=output/models/model_MLM_${date} \
--do_eval \
--do_train \
--model_name_or_path=microsoft/codebert-base \
--overwrite_output_dir \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=32 \
--logging_steps=20 \
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
--load_best_model_at_end=True \
--fp16

