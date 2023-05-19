python3 structgpt_for_tableqa.py \
--api_key ./api_key.txt --num_process 37 \
--prompt_path ./prompts/prompt_for_wikisql.json --prompt_name chat_v1 \
--input_path ./data/wikisql/wikisql_test.json \
--output_path ./outputs/wikisql/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/wikisql/chat_wo_icl_v1.txt --max_tokens 350

cat ./outputs/wikisql/output_wo_icl_v1.jsonl_* >  ./outputs/wikisql/output_wo_icl_v1.jsonl
rm ./outputs/wikisql/output_wo_icl_v1.jsonl_*
cat ./outputs/wikisql/chat_wo_icl_v1.txt_* >  ./outputs/wikisql/chat_wo_icl_v1.txt
rm ./outputs/wikisql/chat_wo_icl_v1.txt_*

python evaluate_for_wikisql.py --ori_path ./data/wikisql/wikisql_test.json --inp_path ./outputs/wikisql/output_wo_icl_v1.jsonl