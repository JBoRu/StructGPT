#!/usr/bin/env bash

python3 structgpt_for_spider.py \
--api_key ./api_key.txt --num_process 21 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./data/spider/dev.jsonl \
--output_path ./outputs/spider/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/spider/chat_wo_icl_v1.txt \
--db_path ./data/spider/all_tables_content.json \
--schema_path ./data/spider/tables.json

# single process usage
#python3 structgpt_for_spider.py \
#--api_key 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 --num_process 1 \
#--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
#--input_path ./data/spider/dev.jsonl \
#--output_path ./outputs/spider/output_wo_icl_v1.jsonl \
#--chat_log_path ./outputs/spider/chat_wo_icl_v1.txt \
#--schema_path ./data/spider/tables.json

cat ./outputs/spider/output_wo_icl_v1.jsonl_* > ./outputs/spider/output_wo_icl_v1.jsonl
rm ./outputs/spider/output_wo_icl_v1.jsonl_*
cat ./outputs/spider/chat_wo_icl_v1.txt_* > ./outputs/spider/chat_wo_icl_v1.txt
rm ./outputs/spider/chat_wo_icl_v1.txt_*

python evaluate_for_spider.py --path ./outputs/spider/output_wo_icl_v1.jsonl --db=data/spider/database --table=data/spider/tables.json --etype=exec