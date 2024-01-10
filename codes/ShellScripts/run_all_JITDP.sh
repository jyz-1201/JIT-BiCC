export PYTHONPATH=$PYTHONPATH:./

date=0425
model_parent_dir="output/models"
multitask_model_name=(model_MLM_0130 model_RNMI_on_MLM_0130_0219_2 model_RNMI_on_CodeBERT_0226
model_RNMI_on_MLM_0130_0303 model_RDNMI_on_MLM_0130_0304 model_RDNMI_on_CodeBERT_0312)


for ((i=0; i<${#multitask_model_name[*]}; i++))
do
  output_dir="${model_parent_dir}/model_JITDP_with_expert_on${multitask_model_name[i]}_${date}"

  python JITFine/concat/run.py \
  --output_dir="${output_dir}"\
  --config_name=microsoft/codebert-base \
  --model_name_or_path="${model_parent_dir}/${multitask_model_name[i]}" \
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
  --buggy_line_filepath=data/jitfine/changes_complete_buggy_line_level.pkl \
  |tee "${output_dir}/train.log"
done

