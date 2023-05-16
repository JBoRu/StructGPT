#!/usr/bin/env bash

python3 structgpt_for_spider.py \
--api_key sk-?? --num_process 1 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat \
--input_path ./data/spider/dev.jsonl \
--output_path ./outputs/spider/output_v2.jsonl \
--chat_log_path ./outputs/spider/chat_log_v2.txt \
--db_path ./data/spider/all_tables_content.json \
--schema_path ./data/spider/tables.json

cat ./outputs/spider/output_v2.jsonl_* > ./outputs/spider/output_v2.jsonl
rm ./outputs/spider/output_v2.jsonl_*
cat ./outputs/spider/chat_log_v2.txt_* > ./outputs/spider/chat_log_v2.txt
rm ./outputs/spider/chat_log_v2.txt_*