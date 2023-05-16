python3 structgpt_for_tabfact.py \
--api_key sk-?? --num_process 1 \
--prompt_path ./prompts/prompt_for_tab_fact.json --prompt_name chat \
--input_path ./data/tabfact/tab_fact_test.json \
--output_path ./outputs/tabfact/output_v2.jsonl \
--chat_log_path ./outputs/tabfact/chat_log_v2.txt \
--log_path ./outputs/tabfact/log_v2.txt --max_tokens 350

cat ./outputs/tabfact/output_v2.jsonl_* >  ./outputs/tabfact/output_v2.jsonl
rm ./outputs/tabfact/output_v2.jsonl_*

cat ./outputs/tabfact/log_v2.txt_* >  ./outputs/tabfact/log_v2.txt
rm ./outputs/tabfact/log_v2.txt_*

cat ./outputs/tabfact/chat_log_v2.txt_* >  ./outputs/tabfact/chat_log_v2.txt
rm ./outputs/tabfact/chat_log_v2.txt_*