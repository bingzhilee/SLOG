archive_path=$1

datapath="data/cogs_LF/gen.json"

python evaluate.py \
    --load_8bit \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights $archive_path \
    --datapath $datapath \
    --pred_output_path $archive_path'/out.test.pred.tsv' \
    --prompt_template 'cogs'