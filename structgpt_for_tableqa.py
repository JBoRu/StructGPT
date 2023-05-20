import argparse
import json
import logging
import os
import pickle
import re
import multiprocessing as mp
from tqdm import tqdm
import time

import openai


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

    def get_response_v1(self, input_text, turn_type):
        message = self.create_message_v1(input_text, turn_type)
        self.history_contents.append(message['content'])
        self.history_messages.append(message)
        message = self.query_API_to_get_message(self.history_messages)
        self.history_contents.append(message['content'])
        self.history_messages.append(message)
        response = message['content']
        return response

    def create_message_v1(self, input_text, turn_type):
        if turn_type == "columns_select":
            template = self.prompt['columns_select']
            columns, question = input_text
            # question = question.capitalize()
            input_text = template.format(question=question, columns=columns)
        elif turn_type == 'rows_select':
            template = self.prompt['rows_select']
            selected_cols, rows, question = input_text
            # question = question.capitalize()
            input_text = template.format(selected_columns=selected_cols, rows=rows, question=question)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_table = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(table=serialized_table, question=question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
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

    def parse_result(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["initial", "question_template"]:
            if "should be" in content:
                content = content.split("should be")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        print("Not exactly parse, we directly use content: %s" % content)

        return content

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class Retriever:
    def __init__(self, args):
        self.args = args

    def serialize_headers(self, headers):
        # headers = ['"' + header.replace("\n", " ") + '"' for header in headers]
        headers = [header.replace("\n", " ") for header in headers]
        # if len(headers) == 0:
        #     ser_hea = ""
        # elif len(headers) == 1:
        #     ser_hea = headers[0]
        # elif len(headers) == 2:
        #     ser_hea = headers[0] + " and " + headers[1]
        # else:
        #     ser_hea = ", ".join(headers[0:-1]) + ", and " + headers[-1]
        ser_hea = ", ".join(headers)
        return ser_hea

    def filter_table_with_col_name(self, table, selected_relations_list, selected_relations_str):
        new_table = dict()
        header = table['header']
        rows = table['rows']
        reserved_col_idx = [idx for idx, rel in enumerate(header) if rel.replace("\n", " ") in selected_relations_list
                            or rel.replace("\n", " ").lower() in selected_relations_str.lower()]
        new_header = [header[idx] for idx in reserved_col_idx]
        new_rows = [[row[idx] for idx in reserved_col_idx] for row in rows]
        new_table["header"] = new_header
        new_table["rows"] = new_rows
        return new_table


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.SLM = Retriever(args)
        self.max_serialization_tokens = args.max_llm_input_tokens
        self.selected_relations = []

    def forward(self, question, table):
        self.LLM.reset_history()
        self.reset_history()

        iterative_step = 0

        # select
        table = self.normalize_table_header(table)
        header = table['header']
        ser_hea = self.SLM.serialize_headers(header)
        if args.debug:
            print("Step-%d: ser_hea:%s" % (iterative_step, ser_hea))

        llm_selected_cols = self.LLM.get_response_v1((ser_hea, question), "columns_select")
        self.LLM.reset_history_messages()
        if args.debug:
            print("Step-%d: llm_selected_cols:%s" % (iterative_step, llm_selected_cols))

        selected_cols_list = self.parse_selected_cols(llm_selected_cols, header)
        if args.debug:
            print("Step-%d: selected_cols_list:%s" % (iterative_step, selected_cols_list))

        filtered_table = self.SLM.filter_table_with_col_name(table, selected_cols_list, llm_selected_cols)
        if args.debug:
            print("Step-%d: filtered_table:%s" % (iterative_step, filtered_table))

        candidate_rows = self.serialize_table(filtered_table)
        if args.debug:
            print("Step-%d: candidate_rows:%s" % (iterative_step, candidate_rows))

        selected_columns = self.SLM.serialize_headers(selected_cols_list)
        choose_rows = self.LLM.get_response_v1((selected_columns, candidate_rows, question), "rows_select")
        self.LLM.reset_history_messages()
        try:
            choose_row_list = self.parse_row_idx(choose_rows)
            filtered_table = self.filter_table_with_rows_constraints(filtered_table, choose_row_list)
        except Exception as e:
            logging.exception(e)
            # print(candidate_rows)
            # print(choose_rows)
        if args.debug:
            print("Step-%d: filtered_table:%s" % (iterative_step, filtered_table))

        serialized_table = self.serialize_table(filtered_table)
        if args.debug:
            print("Step-%d: serialized_table:%s" % (iterative_step, serialized_table))

        final_answers = self.LLM.get_response_v1((question, serialized_table),
                                                         "ask_final_answer_or_next_question")
        self.LLM.reset_history_messages()

        return final_answers, self.LLM.history_contents, self.log

    def is_end(self, response, iterative_step):
        if "no" in response.lower() or iterative_step > 8:
            return True
        else:
            return False

    def is_end_v1(self, response, iterative_step):
        if "final" in response.lower() or iterative_step > 3:
            return True
        elif "next" in response.lower():
            return False
        else:
            return False

    def get_final_answers(self, history_responses, final_response):
        answer_tmp_1 = history_responses[-2]
        answer_tmp_2 = final_response
        return [answer_tmp_1, answer_tmp_2]

    def parse_result(self, response, parse_type):
        response = response.lower()
        if parse_type == "next_question":
            if "the next question:" in response:
                next_question = response.split("the next question:")[1].strip()
            elif ":" in response:
                next_question = response.split(":")[1].strip()
            else:
                next_question = response
                print("Not parse the next question exactly, directly use the response: ", response)
            return next_question
        elif parse_type == "final_answer":
            if 'yes' in response and  'no' in response:
                final_answer = 'unknown'
            elif 'yes' in response:
                final_answer = 'entailed'
            else:
                final_answer='refuted'

            return final_answer

    def reset_history(self):
        self.log = []
        self.selected_relations = []

    def serialize_table(self, table):
        header = table['header']
        rows = table['rows']
        lines = []
        for idx, row in enumerate(rows):
            pairs = []
            for rel, ent in zip(header, row):
                pair = "(" + rel + ", " + ent + ")"
                pairs.append(pair)

            line = 'item ' + str(idx + 1) + ': ' + "; ".join(pairs)
            lines.append(line)
        output = "\n".join(lines)
        return output

    def parse_selected_cols(self, llm_selected_cols, header):
        llm_selected_cols = [h for h in header if h.replace("\n", " ").lower() in llm_selected_cols.lower()]
        return llm_selected_cols

    def parse_row_idx(self, selected_rows):
        pattern = re.compile(r'(\d+)')
        m = pattern.finditer(selected_rows)
        m = [i.group() for i in m]
        selected_rows = [int(rid)-1 for rid in m]
        return selected_rows

    def filter_table_with_rows_constraints(self, table, row_constraints):
        new_table = dict()
        header = table['header']
        rows = table['rows']
        new_rows = []
        for rid in row_constraints:
            if rid < len(rows):
                new_rows.append(rows[rid])
        new_table["header"] = header
        new_table["rows"] = new_rows
        return new_table

    def normalize_table_header(self, table):
        header = table['header']
        rows = table['rows']
        new_table = {}
        new_header = []
        for h in header:
            h = h.replace("\n", " ")
            new_header.append(h)
        new_table['header'] = new_header
        new_table['rows'] = rows
        return new_table

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
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data), desc="PID: %d" % os.getpid()):
                try:
                    question = sample["statement"] if 'statement' in sample else sample['question']
                    question = question + "?" if not question.endswith("?") else question
                    # question = question.capitalize()
                    table = sample['table']
                    prediction, chat_history, record = solver.forward(question, table)
                except openai.error.InvalidRequestError as e:
                    print(e)
                    continue
                except Exception as e:
                    logging.exception(e)
                    continue
                if 'id' in sample.keys():
                    flag = str(sample['id'])
                else:
                    flag = question

                try:
                    chat = flag + "\n" + "\n******\n".join(chat_history) + "\nAnswers: " + str(
                        sample['seq_out']) + "\n------------------------------------------\n"
                    fclog.write(chat)
                except Exception as e:
                    print(e)
                count += 1
                if count < 5:
                    print(sample['seq_out'])
                    print(prediction)
                    print("---------------------")
                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--chat_log_path', default=None)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--prompt_path')
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    parser.add_argument('--max_tokens', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--api_key', default="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", type=str)
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

    with open(args.input_path, "rb") as f:
        all_data = json.load(f)
        print("Totally %d test examples." % len(all_data))

    # Test data that has not yet been predicted
    # if os.path.exists(args.output_path):
    #     with open(args.output_path, "r") as f:
    #         all_lines = f.readlines()
    #         all_lines = [json.loads(line.strip("\n")) for line in all_lines]
    #         already_id = [line['id'] for line in all_lines]
    #         all_data = [data for data in all_data if data['id'] not in already_id]
    #         print("There are %d test examples need to be processed." % len(all_data))

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
            p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
        p.close()
        p.join()
        print("All of the child processes over!")
