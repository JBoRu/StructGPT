#!/usr/bin/env bash

python3 structgpt_for_text_to_sql.py \
--api_key ./api_key.txt --num_process 21 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./data/spider-realistic/spider-realistic.json \
--output_path ./outputs/spider-realistic/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/spider-realistic/chat_wo_icl_v1.txt \
--schema_path ./data/spider-realistic/tables.json

# single process usage
#python3 structgpt_for_text_to_sql.py \
#--api_key sk-?? --num_process 1 \
#--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
#--input_path ./data/spider-realistic/spider-realistic.json \
#--output_path ./outputs/spider-realistic/output_wo_icl_v1.jsonl \
#--chat_log_path ./outputs/spider-realistic/chat_wo_icl_v1.txt \
#--schema_path ./data/spider-realistic/tables.json

cat ./outputs/spider-realistic/output_wo_icl_v1.jsonl_* > ./outputs/spider-realistic/output_wo_icl_v1.jsonl
rm ./outputs/spider-realistic/output_wo_icl_v1.jsonl_*
cat ./outputs/spider-realistic/chat_wo_icl_v1.txt_* > ./outputs/spider-realistic/chat_wo_icl_v1.txt
rm ./outputs/spider-realistic/chat_wo_icl_v1.txt_*

python evaluate_for_spider.py --path ./outputs/spider-realistic/output_wo_icl_v1.jsonl --db data/spider/database --table data/spider-realistic/tables.json --etype exec