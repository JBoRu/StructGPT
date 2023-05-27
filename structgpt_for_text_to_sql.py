import argparse
import json
import logging
import os
import random
from collections import defaultdict
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import time

import openai

random.seed(42)


class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.idx_mapping = {"0": "first", "1": "second", "2": "third", "3": "fourth", "4": "fifth", "5": "sixth",
                            "6": "seventh",
                            "7": "eighth", "8": "ninth", "9": "tenth"}

    def get_response_v1(self, input_text, turn_type, max_output_len):
        message = self.create_message_v1(input_text, turn_type)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        message = self.query_API_to_get_message(self.history_messages, max_output_len)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        response = self.parse_result(message)

        return response

    def create_message_v1(self, input_text, turn_type):
        if turn_type == "select_tab":
            template = self.prompt['free_generate']
            question, ser_table_name = input_text
            input_text = template.format(question=question, table=ser_table_name)
        elif turn_type == "reorg_sel_tab":
            template = self.prompt['table_column_select_reorganize']
            input_text = template
        elif turn_type == "ask_final_answers":
            question, ser_table_name, ser_fks = input_text
            if len(ser_fks) > 1:
                template = self.prompt['ask_final_answers']['has_fk']
                input_text = template.format(question=question, table=ser_table_name, fk=ser_fks)
            else:
                template = self.prompt['ask_final_answers']['no_fk']
                input_text = template.format(question=question, table=ser_table_name)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages, max_output_len):
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=max_output_len,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return res['choices'][0]['message']
            except openai.error.RateLimitError as e:
                err_mes = str(e)
                if "You exceeded your current quota" in err_mes:
                    print("You exceeded your current quota: %s" % openai.api_key)
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(30)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
            except openai.error.InvalidRequestError as e:
                logging.exception(e)
                exit(0)


    def parse_result(self, result):
        content = result['content'].strip()

        return content

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def reseta_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class Retriever:
    def __init__(self, args):
        self.args = args
        self.prompt = self.load_prompt_template(args.prompt_path, args.prompt_name)
        self.creatiing_schema(args.schema_path)

    def filter_table_with_col_name(self, table, selected_relations_list, selected_relations_str):
        new_table = dict()
        header = table['header']
        rows = table['rows']
        reserved_col_idx = [idx for idx, h in enumerate(header) if h in selected_relations_list]
        new_header = [header[idx] for idx in reserved_col_idx]
        new_rows = [[row[idx] for idx in reserved_col_idx] for row in rows]
        new_table["header"] = new_header
        new_table["rows"] = new_rows
        return new_table

    def load_db(self, db_path):
        with open(db_path, "r") as f:
            db = json.load(f)
        return db

    def serialize_table_and_column(self, db_name):
        df = self.spider_schema[self.spider_schema['Database name'] == db_name]
        df = df.groupby('Table Name')
        table2columns = {}
        tables_name = []
        for name, group in df:
            columns = []
            for index, row in group.iterrows():
                columns.append(row["Field Name"])
            table2columns[name] = columns
            if name not in tables_name:
                tables_name.append(name)
        ser_heas = []
        prompt_tmp = "# {tn}({cols});"
        for name, cols in table2columns.items():
            cols = [col for col in cols]
            cols = ",".join(cols)
            hea = prompt_tmp.format(tn=name, cols=cols)
            ser_heas.append(hea)
        ser_hea = "\n".join(ser_heas)
        return ser_hea, table2columns

    def serialize_tab_and_col_of_demons(self, tables):
        prompt_tmp = " # {tn}({cols});"
        ser_heas = []
        for name, cols in tables.items():
            cols = [col for col in cols]
            cols = ",".join(cols)
            hea = prompt_tmp.format(tn=name, cols=cols)
            ser_heas.append(hea)
        ser_heas = "\n".join(ser_heas)
        return ser_heas

    def serialize_selected_table_name(self, sel_tab_cols):
        ser_heas = []
        prompt_tmp = "# {tn}({cols});"
        for name, cols in sel_tab_cols.items():
            cols = [col for col in cols]
            cols = ",".join(cols)
            hea = prompt_tmp.format(tn=name, cols=cols)
            ser_heas.append(hea)

        ser_hea = "\n".join(ser_heas)
        return ser_hea

    def creatiing_schema(self, schema_path):
        schema_df = pd.read_json(schema_path)
        schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)
        schema = []
        f_keys = []
        p_keys = []
        for index, row in schema_df.iterrows():
            tables = row['table_names_original']
            col_names = row['column_names_original']
            col_types = row['column_types']
            foreign_keys = row['foreign_keys']
            primary_keys = row['primary_keys']
            for col, col_type in zip(col_names, col_types):
                index, col_name = col
                if index == -1:
                    for table in tables:
                        schema.append([row['db_id'], table, '*', 'text'])
                else:
                    schema.append([row['db_id'], tables[index], col_name, col_type])
            for primary_key in primary_keys:
                index, column = col_names[primary_key]
                p_keys.append([row['db_id'], tables[index], column])
            for foreign_key in foreign_keys:
                first, second = foreign_key
                first_index, first_column = col_names[first]
                second_index, second_column = col_names[second]
                f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
        self.spider_schema = pd.DataFrame(schema, columns=['Database name', 'Table Name', 'Field Name', 'Type'])
        self.spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
        self.spider_foreign = pd.DataFrame(f_keys,
                                           columns=['Database name', 'First Table Name', 'Second Table Name',
                                                    'First Table Foreign Key',
                                                    'Second Table Foreign Key'])

    def find_primary_keys(self, db_name, table_name):
        df = self.spider_primary[self.spider_primary['Database name'] == db_name]
        for index, row in df.iterrows():
            if row['Table Name'] == table_name:
                return row['Primary Key']

    def find_foreign_keys(self, db_name):
        df = self.spider_foreign[self.spider_foreign['Database name'] == db_name]
        output = []
        for index, row in df.iterrows():
            first_tab_fk = (row['First Table Name'], row['First Table Foreign Key'])
            second_tab_fk = (row['Second Table Name'], row['Second Table Foreign Key'])
            output.append((first_tab_fk, second_tab_fk))
        return output

    def serialize_fk_with_sel_table(self, db_id, table_cols):
        prompt_tmp = "{ftk_0}.{ftk_1}={stk_0}.{stk_1}"
        fk = self.find_foreign_keys(db_id)
        ser_fks = []
        for (ftk, stk) in fk:
            if ftk[0] in table_cols and stk[0] in table_cols:
                ser_fk = prompt_tmp.format(ftk_0=ftk[0], ftk_1=ftk[1], stk_0=stk[0], stk_1=stk[1])
                ser_fks.append(ser_fk)
        ser_fks = ",".join(ser_fks)
        ser_fks = ser_fks + ";"
        return ser_fks

    def serialize_demonstrations(self, demonstrations, example_ids):
        prompt = self.prompt["demonstration"]
        prompt_fk = "{ftk_0}.{ftk_1}={stk_0}.{stk_1}"
        all_sel_demons = []
        for name, ids in example_ids.items():
            demons = demonstrations[name]
            sel_demons = [demons[i] for i in ids]
            all_sel_demons.extend(sel_demons)
        ser_demons = []
        for d in all_sel_demons:
            ser_fks = []
            for (ftk, stk) in d['fks']:
                ser_fk = prompt_fk.format(ftk_0=ftk[0], ftk_1=ftk[1], stk_0=stk[0], stk_1=stk[1])
                ser_fks.append(ser_fk)
            ser_fks = ",".join(ser_fks)
            ser_fks = ser_fks + ";"

            tables = d['tables']
            question = d['question']
            query = d['query']
            ser_tables = self.serialize_tab_and_col_of_demons(tables)

            if len(d['fks']) == 0:
                p = prompt['no_fk']
                one_demo = p.format(table=ser_tables, question=question, sql=query)
            else:
                p = prompt['has_fk']
                one_demo = p.format(table=ser_tables, question=question, sql=query, fk=ser_fks)
            ser_demons.append(one_demo)
        random.shuffle(ser_demons)
        ser_demons = "\n #\n".join(ser_demons)
        return ser_demons

    def serialize_demonstrations_with_const(self, demonstrations, example_ids, num_tables, fk_flag):
        num_tables = 1 if num_tables == 1 else 2
        prompt = self.prompt["demonstration"]
        prompt_fk = "{ftk_0}.{ftk_1}={stk_0}.{stk_1}"
        all_sel_demons = []
        for name, ids in example_ids.items():
            demons = demonstrations[name]
            sel_demons = [demons[i] for i in ids]
            all_sel_demons.extend(sel_demons)
        ser_demons = []
        for d in all_sel_demons:
            ser_fks = []
            for (ftk, stk) in d['fks']:
                ser_fk = prompt_fk.format(ftk_0=ftk[0], ftk_1=ftk[1], stk_0=stk[0], stk_1=stk[1])
                ser_fks.append(ser_fk)
            ser_fks = ",".join(ser_fks)
            ser_fks = ser_fks + ";"

            tables = d['tables']
            num_tables_demon = 1 if len(tables) == 1 else 2

            if num_tables_demon != num_tables:
                continue

            question = d['question']
            query = d['query']
            ser_tables = self.serialize_table_name_v6(tables)

            if len(d['fks']) == 0:
                p = prompt['no_fk']
                one_demo = p.format(table=ser_tables, question=question, sql=query)
            else:
                p = prompt['has_fk']
                one_demo = p.format(table=ser_tables, question=question, sql=query, fk=ser_fks)
            ser_demons.append(one_demo)
        if len(ser_demons) == 0:
            print("-------------------No demonstrations-------------------")
        random.shuffle(ser_demons)
        ser_demons = "\n\n".join(ser_demons)
        return ser_demons

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.SLM = Retriever(args)
        self.max_serialization_tokens = args.max_llm_input_tokens
        self.log = []
        self.selected_relations = []

    def forward_wo_icl_v1(self, question, db_id):
        self.LLM.reset_history()
        self.reset_history()

        iterative_step = 0

        ser_tab_col, table2columns = self.SLM.serialize_table_and_column(db_id)
        if args.debug:
            print("-----Step-%d: ser_table_name:\n%s" % (iterative_step, ser_tab_col))

        llm_select_tab = self.LLM.get_response_v1((question, ser_tab_col), "select_tab", max_output_len=600)
        if args.debug:
            print("-----Step-%d: llm_select_tab:\n%s" % (iterative_step, llm_select_tab))

        llm_reorg_sel_tab = self.LLM.get_response_v1("", "reorg_sel_tab", max_output_len=300)
        if args.debug:
            print("-----Step-%d: llm_reorg_sel_tab:\n%s" % (iterative_step, llm_reorg_sel_tab))
        self.LLM.reset_history_messages()

        sel_tab_cols = self.parse_sele_tab(llm_reorg_sel_tab, table2columns)
        if args.debug:
            print("-----Step-%d: sel_tab_cols:\n%s" % (iterative_step, sel_tab_cols))

        ser_table_name = self.SLM.serialize_selected_table_name(sel_tab_cols)
        ser_fk = self.SLM.serialize_fk_with_sel_table(db_id, sel_tab_cols)
        if args.debug:
            print("-----Step-%d: ser_table_name:\n%s" % (iterative_step, ser_table_name))
            print("-----Step-%d: ser_fk:\n%s" % (iterative_step, ser_fk))

        final_answers = self.LLM.get_response_v1((question, ser_table_name, ser_fk), "ask_final_answers",
                                              max_output_len=300)
        if args.debug:
            print("-----Step-%d: final_answers:\n%s" % (iterative_step, final_answers))
        self.LLM.reset_history_messages()

        return final_answers, self.LLM.history_contents

    def reset_history(self):
        self.log = []
        self.selected_relations = []

    def serialize_table(self, db_name, table_name):
        # get primary key
        pk = self.SLM.find_primary_keys(db_name, table_name)
        # get table content
        table = self.SLM.db_content[db_name][table_name]
        # serialize
        header = table['headers']
        rows = table['rows']
        lines = []
        for idx, row in enumerate(rows):
            pairs = []
            row_name = ""
            for rel, ent in zip(header, row):
                if rel == pk:
                    row_name = pk + " " + str(ent) + ": "
                else:
                    pair = "(" + rel + ", " + str(ent) + ")"
                    pairs.append(pair)
            line = row_name + "; ".join(pairs)
            lines.append(line)
        output = "\n".join(lines)
        return output

    def serialize_table_with_constraints(self, db_name, table_name, constraints_cols):
        # get primary key
        pk = self.SLM.find_primary_keys(db_name, table_name)
        # get table content
        table = self.SLM.db_content[db_name][table_name]
        # serialize
        header = table['headers']
        rows = table['rows']
        lines = []
        if len(constraints_cols) == 1 and pk in constraints_cols:
            constraints_cols.append(random.sample(list(set(header) - set(constraints_cols)), k=1)[0])
        for idx, row in enumerate(rows):
            pairs = []
            row_name = ""
            for rel, ent in zip(header, row):
                if rel == pk:
                    row_name = pk + " " + str(ent) + ": "
                else:
                    if rel in constraints_cols:
                        pair = "(" + rel + ", " + str(ent) + ")"
                        pairs.append(pair)
            line = row_name + "; ".join(pairs)
            lines.append(line)
        output = "\n".join(lines)
        return output

    def parse_sele_tab(self, llm_selected_table_col, table2columns):
        sel_table_to_cols = defaultdict(list)
        try:
            tabs = llm_selected_table_col.strip(" \n|.;`").strip()
            tabs = tabs.split("|")
            tabs = [tab.strip(" \n|.;`") for tab in tabs]
            for tab in tabs:
                if tab in table2columns:
                    sel_table_to_cols[tab] = table2columns[tab]
        except Exception as e:
            logging.exception(e)
            for tab, cols in table2columns.items():
                if tab in llm_selected_table_col:
                    sel_table_to_cols[tab] = table2columns[tab]
            print("*****LLM selected tables output doesn't match the predefined format:\n%s" % llm_selected_table_col)
        return sel_table_to_cols


def main(args, all_data, idx, api_key):
    import openai
    openai.api_key = api_key

    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_path))
    solver = Solver(args)
    
    count = 0
    valid_count = 0
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data), desc="PID: %d" % os.getpid()):
                try:
                    if "question" in sample:
                        question = sample["question"]
                    elif "SpiderSynQuestion" in sample:
                        question = sample["SpiderSynQuestion"]
                    else:
                        print("Specify an error question key.")
                        print(sample)
                        exit(0)
                    db_id = sample['db_id']

                    if not args.icl:
                        results = solver.forward_wo_icl_v1(question, db_id)

                    if results is not None:
                        prediction, chat_history = results
                        valid_count += 1
                    else:
                        continue
                except Exception as e:
                    logging.exception(e)
                    continue

                if 'id' in sample:
                    flag = sample['id']
                else:
                    flag = question
                chat = flag + "\n" + "\n******\n".join(chat_history) + \
                       "\nGold SQL: " + str(sample['query']) + "\n------------------------------------------\n"
                fclog.write(chat)

                count += 1
                if count < 3:
                    print(prediction)
                    print("---------------------")
                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")
    print("---------------PID %d end with %d/%d samples--------------" % (os.getpid(), valid_count, count))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--chat_log_path', default=None)
    parser.add_argument('--schema_path', default=None)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--prompt_path')
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--icl', action="store_true", )
    parser.add_argument('--demonstrations', default=None)
    parser.add_argument('--example_ids', default=None)
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    parser.add_argument('--max_tokens', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--max_llm_input_tokens', default=3400, type=int)

    args = parser.parse_args()

    print("Start querying the LLM.")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)

    if args.input_path.endswith('jsonl'):
        with open(args.input_path, "r") as f:
            all_lines = f.readlines()
            all_data = [json.loads(line) for line in all_lines]
            print("Totally %d test examples." % len(all_data))
    elif args.input_path.endswith('json'):
        with open(args.input_path, "r") as f:
            all_data = json.load(f)
            print("Totally %d test examples." % len(all_data))

    if args.demonstrations is not None:
        with open(args.demonstrations, "r") as f:
            demonstrations = json.load(f)
    if args.example_ids is not None:
        with open(args.example_ids, "r") as f:
            example_ids = json.load(f)
            
    if args.num_process == 1:
        main(args, all_data, idx=-1, api_key=args.api_key)
    else:
        num_each_split = int(len(all_data) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(all_data))
            else:
                end = (idx + 1) * num_each_split
            split_data = all_data[start:end]
            try:
                p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
            except Exception as e:
                logging.exception(e)
        p.close()
        p.join()
        print("All of the child processes over!")
