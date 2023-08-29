
archive_dir='model_archives/cogs_LF/llama-7b-hf'
mkdir -p $archive_dir

seed=$1

python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'data/cogs_LF/train.json' \
    --output_dir "${archive_dir}/${seed}" \
    --num_epochs=10 \
    --lora_target_modules='[q_proj,v_proj]' \
    --batch_size 64 \
    --micro_batch_size 64 \
    --val_set_size 0 \
    --learning_rate 3e-4 \
    --train_on_inputs False \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed $seed