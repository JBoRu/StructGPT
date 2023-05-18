#!/usr/bin/env bash

python3 structgpt_for_spider.py \
--api_key ./api_key.txt --num_process 21 \
--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
--input_path ./data/spider-syn/dev.json \
--output_path ./outputs/spider-syn/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/spider-syn/chat_wo_icl_v1.txt \
--schema_path ./data/spider-syn/tables.json

# single process usage
#python3 structgpt_for_spider.py \
#--api_key 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 --num_process 1 \
#--prompt_path ./prompts/prompt_for_spider.json --prompt_name chat_v1 \
#--input_path ./data/spider-syn/spider-syn.json \
#--output_path ./outputs/spider-syn/output_wo_icl_v1.jsonl \
#--chat_log_path ./outputs/spider-syn/chat_wo_icl_v1.txt \
#--schema_path ./data/spider-syn/tables.json

cat ./outputs/spider-syn/output_wo_icl_v1.jsonl_* > ./outputs/spider-syn/output_wo_icl_v1.jsonl
rm ./outputs/spider-syn/output_wo_icl_v1.jsonl_*
cat ./outputs/spider-syn/chat_wo_icl_v1.txt_* > ./outputs/spider-syn/chat_wo_icl_v1.txt
rm ./outputs/spider-syn/chat_wo_icl_v1.txt_*

python evaluate_for_spider.py --path ./outputs/spider-syn/output_wo_icl_v1.jsonl --db data/spider/database --table data/spider-syn/tables.json --etype exec