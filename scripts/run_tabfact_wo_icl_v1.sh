python3 structgpt_for_tableqa.py \
--api_key ./api_key.txt --num_process 37 \
--prompt_path ./prompts/prompt_for_tabfact.json --prompt_name chat_v1 \
--input_path ./data/tabfact/tab_fact_test.json \
--output_path ./outputs/tabfact/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/tabfact/chat_wo_icl_v1.txt --max_tokens 350

cat ./outputs/tabfact/output_wo_icl_v1.jsonl_* >  ./outputs/tabfact/output_wo_icl_v1.jsonl
rm ./outputs/tabfact/output_wo_icl_v1.jsonl_*
cat ./outputs/tabfact/chat_wo_icl_v1.txt_* >  ./outputs/tabfact/chat_wo_icl_v1.txt
rm ./outputs/tabfact/chat_wo_icl_v1.txt_*

python evaluate_for_tabfact.py --ori_path ./data/tabfact/tab_fact_test.json --inp_path ./outputs/tabfact/output_wo_icl_v1.jsonl