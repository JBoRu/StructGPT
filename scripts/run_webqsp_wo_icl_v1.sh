python3 structgpt_for_webqsp.py \
--api_key ./api_key_for_kg.txt --num_process 14 \
--prompt_path ./prompts/prompt_for_webqsp.json --max_tokens 300 --prompt_name chat_v1 \
--kg_source_path ./data/webqsp/subgraph_2hop_triples.npy \
--ent_type_path ./data/webqsp/ent_type_ary.npy \
--ent2id_path ./data/webqsp/ent2id.pickle \
--rel2id_path ./data/webqsp/rel2id.pickle \
--ent2name_path ./data/webqsp/entity_name.pickle \
--max_triples_per_relation 60 \
--input_path ./data/webqsp/webqsp_simple_test.jsonl \
--output_path ./outputs/webqsp/output_wo_icl_v1.jsonl \
--chat_log_path ./outputs/webqsp/chat_wo_icl_v1.txt

cat ./outputs/webqsp/output_wo_icl_v1.jsonl_* >  ./outputs/webqsp/output_wo_icl_v1.jsonl
rm ./outputs/webqsp/output_wo_icl_v1.jsonl_*
cat ./outputs/webqsp/chat_wo_icl_v1.txt_* >  ./outputs/webqsp/chat_wo_icl_v1.txt
rm ./outputs/webqsp/chat_wo_icl_v1.txt_*