python3 structgpt_for_tableqa.py \
--api_key ./api_key.txt --num_process 37 \
--prompt_path ./prompts/prompt_for_wtq.json --prompt_name chat_v1 \
--input_path ./data/wtq/wikitq_test.json \
--output_path ./outputs/wtq/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/wtq/chat_wo_icl_v1.txt --max_tokens 350

cat ./outputs/wtq/output_wo_icl_v1.jsonl_* >  ./outputs/wtq/output_wo_icl_v1.jsonl
rm ./outputs/wtq/output_wo_icl_v1.jsonl_*
cat ./outputs/wtq/chat_wo_icl_v1.txt_* >  ./outputs/wtq/chat_wo_icl_v1.txt
rm ./outputs/wtq/chat_wo_icl_v1.txt_*

python evaluate_for_wikisql.py --ori_path ./data/wtq/wikitq_test.json --inp_path ./outputs/wtq/output_wo_icl_v1.jsonl
