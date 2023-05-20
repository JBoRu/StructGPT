import argparse
import json
import logging
import os
import pickle
import re

import openai
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *
import multiprocessing as mp


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

    def get_response(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result(message)
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result(message, turn_type)
        return response

    def get_response_v1(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message_v1(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result(message)
            response = input("input the returned response:")
        else:
            message = self.create_message_v1(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result_v1(message, turn_type)
        return response

    def create_message(self, input_text, turn_type, tpe_name):
        if turn_type == "initial":  # the initial query
            instruction = self.prompt[turn_type]['instruction']
            template = self.prompt[turn_type]['init_template']
            self.question = input_text
            input_text = instruction + template.format(question=input_text, tpe=tpe_name)
        elif turn_type == "continue_template":
            input_text = self.prompt[turn_type]
        elif turn_type == "question_template":
            template = self.prompt[turn_type]
            input_text = template.format(idx=self.idx_mapping[input_text])
        elif turn_type == "answer_template":
            template = self.prompt[turn_type]
            if len(input_text) > 0:
                input_text = template["valid"].format(facts=input_text)
            else:
                input_text = template["invalid"]
        elif turn_type == "final_query_template":
            template = self.prompt[turn_type]
            input_text = template.format(question=self.question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def create_message_v1(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            question, tpe, can_rels = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels)
        elif turn_type == "ask_question":
            template = self.prompt['ask_question']
            idx, relations = input_text
            idx = self.idx_mapping[idx]
            input_text = template.format(idx=idx, relations=relations)
        elif turn_type == "ask_answer":
            facts = input_text
            template = self.prompt['ask_answer']
            input_text = template.format(facts=facts)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "condition":
            input_text = self.prompt['continue_template']['condition']
        elif turn_type == "continue":
            input_text = self.prompt['continue_template']['continue']
        elif turn_type == "stop":
            input_text = self.prompt['continue_template']['stop']
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            question, can_rels = input_text
            input_text = template.format(question=question, relations=can_rels)
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
            except openai.error.RateLimitError:
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
            # except openai.error.InvalidRequestError:
            #     print('openai.error.InvalidRequestError\nRetrying...')

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

    def parse_result_v1(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["ask_question", "continue"]:
            if "the simple question:" in content:
                content = content.split("the simple question:")[1].strip()
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

    def parse_result_v2(self, result, turn_type):
        content = result['content'].strip()

        return content

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]

    def get_response_v2(self, input_text, turn_type):
        message = self.create_message_v2(input_text, turn_type)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        message = self.query_API_to_get_message(self.history_messages)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        response = message['content'].strip()

        return response

    def create_message_v2(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        # ykm
        # elif turn_type == "init_relation_rerank":
        #     template = self.prompt['init_relation_rerank']
        #     can_rels, question, tpe, hop = input_text
        #     if hop == 1:
        #         hop = "first"
        #     elif hop == 2:
        #         hop = "second"
        #     elif hop == 3:
        #         hop = "third"
        #     input_text = template.format(question=question, tpe=tpe, relations=can_rels, hop=hop)
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            can_rels, question, tpe = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels)
        elif turn_type == "constraints_flag":
            template = self.prompt['constraints_flag']
            question, tpe, selected_relations = input_text
            if len(selected_relations) > 1:
                selected_relations = "are " + ", ".join(selected_relations)
            else:
                selected_relations = "is " + ", ".join(selected_relations)
            input_text = template.format(question=question, tpe=tpe, selected_relations=selected_relations)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "choose_constraints":
            question, relation_tails, tpe_name = input_text
            template = self.prompt['choose_constraints']
            input_text = template.format(question=question, relation_tails=relation_tails, tpe=tpe_name)
        elif turn_type == "final_query_template":
            template = self.prompt['final_query_template']
            input_text = template.format(question=input_text)
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            can_rels, question, tpe, selected_relations = input_text
            # 暂时注释掉
            # if len(selected_relations) > 1:
            #     selected_relations = "are " + ", ".join(selected_relations)
            # else:
            #     selected_relations = "is " + ", ".join(selected_relations)
            selected_relations = "".join(selected_relations)
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         selected_relations=selected_relations)
        elif turn_type == 'relation_rerank_2hop':
            template = self.prompt['relation_rerank_2hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            sub_question = ", ".join(sub_question)
            selected_relations = ", ".join(selected_relations)
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=sub_question, first_relation=selected_relations)
        elif turn_type == 'relation_rerank_3hop':
            template = self.prompt['relation_rerank_3hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            first_sub_question = sub_question[0]
            second_sub_question = sub_question[1]
            fisrt_relation = selected_relations[0]
            second_relation = selected_relations[1]
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=first_sub_question, first_relation = fisrt_relation,
                                         second_sub_question=second_sub_question, second_relation=second_relation)
        elif turn_type == 'direct_ask_final_answer':
            template = self.prompt['direct_ask_final_answer']
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == 'final_answer_organize':
            template = self.prompt['final_answer_organize']
            input_text = template
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message


class Retriever:
    def __init__(self, args):
        self.args = args
        # self.initialize_PLM(args)
        self.initialize_KG(args)

    def get_retrieval_information(self, first_flag=False, gold_relations=None):
        triples_per_hop, tails = self.KG.get_facts_1hop(self.cur_ents, self.args.max_triples_per_relation,
                                                        first_flag, gold_relations)
        self.reset_cur_ents(tails)
        # self.reset_last_ents(self.cur_ents)
        return triples_per_hop

    # 直接获得triples
    def get_retrieval_information_direct(self, response, tpe, first_flag=False, gold_relations=None):
        triples, tails = self.KG.get_facts_1hop_direct(response, tpe, self.cur_ents, self.tokenizer, self.retriever,
                                                        self.args.topk,
                                                        self.args.filter_score, self.args.max_triples_per_relation,
                                                        first_flag, gold_relations)
        self.reset_cur_ents(tails)
        # self.reset_last_ents(self.cur_ents)
        return triples

    def get_retrieval_relations(self, first_flag=False):
        rels = self.KG.get_rels_1hop(self.cur_ents, first_flag)
        return rels

    def initialize_KG(self, args):
        self.KG = KnowledgeGraph(args.kg_source_path, args.ent_type_path, args.ent2id_path, args.rel2id_path)

    def initialize_PLM(self, args):
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer = tokenizer
        model = AutoModel.from_pretrained(args.model_path)
        # self.retriever = model.cuda("cuda:" + str(args.device))
        self.retriever = None

    def reset_cur_ents(self, entity_list):
        self.cur_ents = entity_list
        # print("Current entity num: ", len(self.cur_ents))

    def update_cur_ents(self, filtered_triples_per_hop):
        new_tails = set()
        for tri in filtered_triples_per_hop[1]:
            h, r, t = tri
            try:
                t_id = self.KG.ent2id[t]
                new_tails.add(t_id)
            except Exception as e:
                logging.exception(e)
                print("Entity string: %s not in ent2id dict" % t)
                continue
        new_tails = list(new_tails)
        self.reset_cur_ents(new_tails)

    def extract_facts(self, facts, response):
        response = response.lower().strip()
        # if response.startswith("the relevant relations:"):
        #     response = response.replace("the relevant relations:", "")
        #     response = response.strip()
        #     nor_rels = response.split(",")
        #     nor_rels = [rel.strip() for rel in nor_rels]
        # else:
        #     nor_rels = response

        filtered_facts = []
        for tri in facts:
            h, r, t = tri
            if self.filter_relation(r):
                continue
            nor_r = self.normalize_relation(r)
            if nor_r in response:
                filtered_facts.append(tri)
        return filtered_facts

    def filter_relation(self, rel):
        # same criteria as GraftNet
        relation = rel
        if relation == "common.topic.notable_types": return False
        if relation == "base.kwebbase.kwtopic.has_sentences": return False
        domain = relation.split(".")[0]
        if domain == "type" or domain == "common": return True
        return False

    def should_ignore(self, rel):
        if self.filter_relation(rel):
            return True
        return False

    def normalize_relation(self, rel):
        # e.g. <fb:film.film.other_crew>
        rel_surface = rel
        # replace '.' and '_' with ' '
        rel_surface = rel_surface.replace('.', ' ')
        # only keep the last two words
        rel_surface = ' '.join(rel_surface.split(' ')[-2:])
        rel_surface = rel_surface.replace('_', ' ')
        return rel_surface

    def get_one_hop_cand_rels(self, question):
        pass

    def get_tails_list(self, cur_ents):
        tails = [self.KG.id2ent[ent] for ent in cur_ents]
        return tails


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.SLM = Retriever(args)
        self.max_serialization_tokens = args.max_llm_input_tokens
        self.load_ent2name(args.ent2name_path)
        self.log = []
        self.selected_relations = []
        # 暂时添加一个selected_sub_questions = []来存放解析的子问题
        self.selected_sub_questions = []


    def forward_v2(self, question, tpe_str, tpe_id):
        self.LLM.reset_history()
        self.SLM.reset_cur_ents([tpe_id])
        self.reset_history()

        iterative_step = 0

        # start_response = self.LLM.get_response_v2("", "instruction")
        # self.log.append(start_response)

        while True:
            # select
            all_rel_one_hop = self.SLM.get_retrieval_relations(first_flag=iterative_step == 0)
            if len(all_rel_one_hop) == 0:
                final_answers = self.LLM.get_response_v2(question, "final_query_template")
                break

            serialized_rels = self.extract_can_rels(all_rel_one_hop, normalize_rel=False)
            if args.debug:
                print("Step-%d: serialized_rels:%s" % (iterative_step, serialized_rels))

            if iterative_step == 0:
                llm_selected_rels = self.LLM.get_response_v2((serialized_rels, question, tpe_str),
                                                             "init_relation_rerank")
            else:
                llm_selected_rels = self.LLM.get_response_v2(
                    (serialized_rels, question, tpe_str, self.selected_relations), "relation_rerank")
            self.LLM.reset_history_messages()
            if args.debug:
                print("Step-%d: llm_selected_rels:%s" % (iterative_step, llm_selected_rels))

            selected_relations_list = self.parse_llm_selected_relations(llm_selected_rels, all_rel_one_hop)
            if args.debug:
                print("Step-%d: selected_relations_list:%s" % (iterative_step, selected_relations_list))
            if len(selected_relations_list) == 0:
                final_answers = self.LLM.get_response_v2(question, "final_query_template")
                break

            self.selected_relations.extend(selected_relations_list)
            if args.debug:
                print("Step-%d: self.selected_relations:%s" % (iterative_step, self.selected_relations))

            filtered_triples_per_hop = self.SLM.get_retrieval_information(first_flag=iterative_step == 0,
                                                                          gold_relations=selected_relations_list)
            cvt_triples, mid_triples, entstr_triples = self.classify_triples(filtered_triples_per_hop)
            if len(cvt_triples) > 0:
                # constraint
                if args.debug:
                    print("Step-%d: Constraints" % iterative_step)
                constraints_candidate = self.serialize_constraints(cvt_triples)
                if args.debug:
                    print("Step-%d: constraints_candidate:%s" % (iterative_step, constraints_candidate))
                constraint_response = self.LLM.get_response_v2((question, constraints_candidate, tpe_str), "choose_constraints")
                self.log.append(constraint_response)
                if args.debug:
                    print("Step-%d: constraint_response:%s" % (iterative_step, constraint_response))
                if self.has_constraints(constraint_response):
                    filtered_triples_per_hop = self.filter_triples(filtered_triples_per_hop, cvt_triples,
                                                                   constraint_response)
                    self.SLM.update_cur_ents(filtered_triples_per_hop)
                    if args.debug:
                        print("Step-%d: filtered_triples_per_hop:%s" % (iterative_step, filtered_triples_per_hop))
                    if args.debug:
                        print("Step-%d: self.SLM.cur_ents:%s" % (iterative_step, self.SLM.cur_ents))
            serialized_facts = self.serialize_facts(filtered_triples_per_hop)
            self.log.append(serialized_facts)

            if args.debug:
                print("Step-%d: serialized_facts:%s" % (iterative_step, serialized_facts))

            final_ans_or_next_que = self.LLM.get_response_v2((question, serialized_facts),
                                                             "ask_final_answer_or_next_question")
            self.log.append(final_ans_or_next_que)

            # 新加的
            final_answers = self.parse_result(final_ans_or_next_que, "final_answer")
            self.log.append(final_answers)
            break
        return final_answers, self.LLM.history_contents, self.log

    def reset_selected_list(self):
        self.selected_sub_questions = []
        self.selected_relations = []

    def is_end(self, response, iterative_step):
        if "no" in response.lower() or iterative_step > 8:
            return True
        else:
            return False

    def load_ent2name(self, ent2name_path):
        with open(ent2name_path, "rb") as f:
            self.cvt_flag_dict, self.mid_mapping_dict = pickle.load(f)

    def convert_hyper_facts_to_text(self, facts):
        subj, rels, objs = facts

        if self.is_cvt(subj):
            return None
        elif subj in self.mid_mapping_dict:
            subj_surface = self.mid_mapping_dict[subj]
        elif self.is_ent(subj):
            # print("head entity %s doesn't have name, we skip this triple." % subj)
            return None
        else:
            subj_surface = subj

        flat_facts = []
        for rel, obj in zip(rels, objs):
            if self.should_ignore(rel):
                continue
            else:
                nor_rel = self.normalize_relation(rel)

            if self.is_cvt(obj):
                continue
            elif obj in self.mid_mapping_dict:
                obj_surface = self.mid_mapping_dict[obj]
            elif self.is_ent(obj):
                # print("tail entity %s doesn't have name, we skip this triple." % obj)
                continue
            else:
                obj_surface = obj

            flat_facts.append((subj_surface, nor_rel, obj_surface))

        return flat_facts

    def convert_fact_to_text(self, fact, normalize_rel=False):
        subj, rel, obj = fact

        if self.should_ignore(rel):
            return None

        if rel.endswith(".from"):
            rel = rel.rstrip(".from")
            rel = rel + ".start_time"
        if rel.endswith(".to"):
            rel = rel.rstrip(".to")
            rel = rel + ".end_time"
        rel_surface = self.normalize_relation(rel) if normalize_rel else rel

        # subject
        if subj.startswith("CVT"):
            subj_surface = subj
        elif subj in self.mid_mapping_dict:
            subj_surface = self.mid_mapping_dict[subj]
        elif subj.startswith("m.") or subj.startswith('g.'):
            # print("head entity %s doesn't have name, we skip this triple." % subj)
            return None
        else:
            subj_surface = subj

        # object
        if obj.startswith("CVT"):
            obj_surface = obj
        elif obj in self.mid_mapping_dict:
            obj_surface = self.mid_mapping_dict[obj]
        elif obj.startswith("m.") or obj.startswith('g.'):
            # print("tail entity %s doesn't have name, we skip this triple." % obj)
            return None
        else:
            obj_surface = obj

        return (subj_surface, rel_surface, obj_surface)

    def extract_can_rels(self, all_rel_one_hop, normalize_rel=True):
        rel_prompt = '"{relation}"'
        nor_rels_set = []
        for rel in all_rel_one_hop:
            if self.filter_relation(rel):
                continue
            nor_r = self.normalize_relation(rel) if normalize_rel else rel
            if nor_r not in nor_rels_set:
                nor_rels_set.append(rel_prompt.format(relation=nor_r))
        rel_candidate = ", ".join(nor_rels_set)
        return rel_candidate

    def serialize_rels(self, rels, normalize_rel=True):
        nor_rels_set = []
        for rel in rels:
            if self.filter_relation(rel):
                continue
            nor_r = self.normalize_relation(rel) if normalize_rel else rel
            if nor_r not in nor_rels_set:
                nor_rels_set.append(nor_r)
        # rel_candidate = ", ".join(nor_rels_set)
        rel_candidate = ";\n ".join(nor_rels_set)
        return rel_candidate

    # 直接拼接
    def serialize_facts_direct(self, facts):
        # 拼接triples
        facts_str_for_one_tail_ent = ["(" + ", ".join(fact) + ")" for fact in facts]

        serialized_facts = ""
        for fact in facts_str_for_one_tail_ent:
            serialized_facts_tmp = serialized_facts + fact + "; "
            serialized_facts = serialized_facts_tmp
        return serialized_facts

    def serialize_facts(self, facts_per_hop):
        h_r_t = defaultdict(lambda: defaultdict(set))
        visited_flag = {}
        name2cvt_tmp = {}
        cvt_count = 0
        all_facts = []
        for hop, facts in facts_per_hop.items():
            if len(facts) > 0:
                for fact in facts:
                    h, r, t = fact
                    if self.is_cvt(h):
                        if h not in name2cvt_tmp:
                            cvt = "CVT_" + str(cvt_count)
                            cvt_count += 1
                            name2cvt_tmp[h] = cvt
                        h = name2cvt_tmp[h]
                    if self.is_cvt(t):
                        if t not in name2cvt_tmp:
                            cvt = "CVT_" + str(cvt_count)
                            cvt_count += 1
                            name2cvt_tmp[t] = cvt
                        t = name2cvt_tmp[t]
                    fact = (h, r, t)
                    all_facts.append(fact)
                    visited_flag[fact] = False
                    h_r_t[h][r].add(t)

        if len(all_facts) > 0:
            all_facts_str = []
            for tri in all_facts:
                facts_str_for_one_tail_ent = []
                if not visited_flag[tri]:
                    h, r, t = tri
                    if t.startswith("CVT") and len(h_r_t[t]) == 0:
                        continue

                    if h.startswith("CVT"):
                        # print("Qid:[%s] has single cvt head entities." % qid)
                        # logger.info(triples_per_hop)
                        continue
                    elif t.startswith("CVT"):
                        st = self.convert_fact_to_text(tri, normalize_rel=False)
                        facts_str_for_one_tail_ent.append(st)
                        one_hop_triples = h_r_t[t]
                        if len(one_hop_triples) > 0:
                            for key_r, value_ts in one_hop_triples.items():
                                for t_ in value_ts:
                                    visit_tri = (t, key_r, t_)
                                    if not visited_flag[visit_tri]:
                                        visited_flag[visit_tri] = True
                                        st = self.convert_fact_to_text(visit_tri, normalize_rel=False)
                                        if st is not None:
                                            assert len(st) == 3
                                            facts_str_for_one_tail_ent.append(st)
                            # h_new = t
                            # r_new = []
                            # t_new = []
                            # for key_r, value_ts in one_hop_triples.items():
                            #     for t_ in value_ts:
                            #         visit_tri = (t, key_r, t_)
                            #         if not visited_flag[visit_tri]:
                            #             r_new.append(key_r)
                            #             t_new.append(t_)
                            #             visited_flag[visit_tri] = True
                            # tri_new = (t, r_new, t_new)
                            # if len(r_new) == len(t_new) > 0:
                            #     str_tri_list = self.convert_hyper_facts_to_text(tri_new)
                            #     if str_tri_list is not None:
                            #         for st in str_tri_list:
                            #             assert len(st) == 3
                            #             if st not in facts_str:
                            #                 facts_str.append(st)
                    else:
                        st = self.convert_fact_to_text(tri, normalize_rel=False)
                        if st is not None:
                            assert len(st) == 3
                            if st not in facts_str_for_one_tail_ent:
                                facts_str_for_one_tail_ent.append(st)
                facts_str_for_one_tail_ent = ["(" + ", ".join(fact) + ")" for fact in facts_str_for_one_tail_ent]
                facts_str = ", ".join(facts_str_for_one_tail_ent)
                all_facts_str.append(facts_str)

            # facts_str = ["(" + ", ".join(fact) + ")" for fact in facts_str]
            serialized_facts = ""
            for fact in all_facts_str:
                serialized_facts_tmp = serialized_facts + fact + "; "
                if len(serialized_facts_tmp.split()) > self.max_serialization_tokens:
                    break
                else:
                    serialized_facts = serialized_facts_tmp
            serialized_facts = serialized_facts.strip("; ")
        else:
            serialized_facts = ""
        return serialized_facts

    def serialize_facts_v1(self, facts):
        if len(facts) > 0:
            h_r_t = defaultdict(lambda: defaultdict(set))
            visited_flag = {}
            for fact in facts:
                h, r, t = fact
                visited_flag[tuple(fact)] = False
                h_r_t[h][r].add(t)
            facts_str = []
            for tri in facts:
                if not visited_flag[tuple(tri)]:
                    h, r, t = tri
                    if self.is_cvt(t) and len(h_r_t[t]) == 0:
                        continue
                    if self.is_cvt(h):
                        # print("Qid:[%s] has single cvt head entities." % qid)
                        # logger.info(triples_per_hop)
                        continue
                    elif self.is_cvt(t):
                        one_hop_triples = h_r_t[t]
                        if len(one_hop_triples) > 0:
                            h_new = t
                            r_new = []
                            t_new = []
                            for key_r, value_ts in one_hop_triples.items():
                                for t_ in value_ts:
                                    visit_tri = (t, key_r, t_)
                                    if not visited_flag[visit_tri]:
                                        r_new.append(key_r)
                                        t_new.append(t_)
                                        visited_flag[visit_tri] = True
                            tri_new = (h, r_new, t_new)
                            if len(r_new) == len(t_new) > 0:
                                str_tri_list = self.convert_hyper_facts_to_text(tri_new)
                                if str_tri_list is not None:
                                    for st in str_tri_list:
                                        assert len(st) == 3
                                        if st not in facts_str:
                                            facts_str.append(st)
                    else:
                        st = self.convert_fact_to_text(tri)
                        if st is not None:
                            assert len(st) == 3
                            if st not in facts_str:
                                facts_str.append(st)
            facts_str = ["(" + ", ".join(fact) + ")" for fact in facts_str]
            serialized_facts = ""
            for fact in facts_str:
                serialized_facts_tmp = serialized_facts + fact + "; "
                if len(serialized_facts_tmp.split()) > self.max_serialization_tokens:
                    break
                else:
                    serialized_facts = serialized_facts_tmp
            # serialized_facts = "; ".join(facts_str)
            serialized_facts = serialized_facts.strip("; ")
        else:
            serialized_facts = ""
        return serialized_facts

    def is_cvt(self, entity):
        if self.cvt_flag_dict[entity]:
            return True
        else:
            return False

    def is_ent(self, ent_str):
        if type(ent_str) is not bool and (ent_str.startswith("m.") or ent_str.startswith("g.")):
            return True
        else:
            return False

    def filter_relation(self, rel):
        # same criteria as GraftNet
        relation = rel
        if relation == "common.topic.notable_types": return False
        if relation == "base.kwebbase.kwtopic.has_sentences": return False
        domain = relation.split(".")[0]
        if domain == "type" or domain == "common": return True
        return False

    def should_ignore(self, rel):
        if self.filter_relation(rel):
            return True
        return False

    def normalize_relation(self, rel):
        # e.g. <fb:film.film.other_crew>
        rel_surface = rel
        # replace '.' and '_' with ' '
        rel_surface = rel_surface.replace('.', ' ')
        # only keep the last two words
        rel_surface = ' '.join(rel_surface.split(' ')[-2:])
        rel_surface = rel_surface.replace('_', ' ')
        return rel_surface

    def parse_llm_selected_relations(self, llm_sel_rels_str, can_rels):
        # llm_sel_rels = llm_sel_rels_str.strip(" ;.|,<>`[]'")
        # llm_sel_rels = llm_sel_rels.split(',')
        # llm_sel_rels = [rel.strip(" ;.|,<>`[]'").strip(" ;.|,<>`[]'") for rel in llm_sel_rels]
        # llm_sel_rel_list = []
        # for rel in llm_sel_rels:
        #     if rel in can_rels:
        #         llm_sel_rel_list.append(rel)
        #     else:
        #         print(rel)
        # if len(llm_sel_rel_list) == 0:
        #     for rel in can_rels:
        #         if rel in llm_sel_rels_str:
        #             llm_sel_rel_list.append(rel)
        #     print("-----llm_ser_rels:\n%s\ndoesn't match the predefined format" % llm_sel_rels)
        llm_sel_rel_list = []
        for rel in can_rels:
            if rel in llm_sel_rels_str:
                llm_sel_rel_list.append(rel)
        return llm_sel_rel_list

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
            if "the final answers:" in response:
                final_answer = response.split("the final answers:")[1].strip()
            # 暂时注释掉
            elif ":" in response:
                final_answer = response.split(":")[1].strip()
            # 新添加的用于解析direct query
            else:
                final_answer = response
                # 暂时注释掉
                # print("Not parse the final answer exactly, directly use the response: ", response)
            return final_answer

    def classify_triples(self, filtered_triples_per_hop):
        cvt_triples, mid_triples, entstr_triples = set(), set(), set()
        if 0 in filtered_triples_per_hop:
            triples_0 = filtered_triples_per_hop[0]
        else:
            triples_0 = []
        if 1 in filtered_triples_per_hop:
            triples_1 = filtered_triples_per_hop[1]
        else:
            triples_1 = []

        if len(triples_1) == 0:
            for tri in triples_0:
                if self.is_ent(tri[2]):
                    mid_triples.add(tuple(tri))
                else:
                    entstr_triples.add(tuple(tri))
        else:
            for tri in triples_1:
                cvt_triples.add(tuple(tri))
        return cvt_triples, mid_triples, entstr_triples

    def serialize_constraints(self, cvt_triples):
        r2t_set = defaultdict(set)
        for tri in cvt_triples:
            subj, rel, obj = tri
            if self.should_ignore(rel):
                continue

            if rel.endswith(".from"):
                rel = rel.rstrip(".from")
                rel = rel + ".start_time"
            if rel.endswith(".to"):
                rel = rel.rstrip(".to")
                rel = rel + ".end_time"

            rel_surface = rel

            # object
            if obj in self.mid_mapping_dict:
                obj_surface = self.mid_mapping_dict[obj]
            elif obj.startswith("m.") or obj.startswith('g.'):
                # print("tail entity %s doesn't have name, we skip this triple." % obj)
                continue
            else:
                obj_surface = obj

            if obj_surface == "To" or "has_no_value" in rel:
                continue

            r2t_set[rel_surface].add(obj_surface)

        constraints = []
        for r, t_set in r2t_set.items():
            t_set = ['"' + t + '"' for t in t_set]
            constraints.append('"' + r + '"' + ": [" + ", ".join(t_set) + "]")
        # constraints = constraints.rstrip("\n")
        constraints = "\n".join(constraints)
        return constraints

    def has_constraints(self, constraint_response):
        if "no" in constraint_response.lower():
            return False
        else:
            return True

    def filter_triples(self, filtered_triples_per_hop, cvt_triples, constraint_response):
        valid_cvt_nodes = set()
        h_r_t = defaultdict(list)
        for tri in cvt_triples:
            h, r, t = tri
            h_r_t[h].append((r, t))
        for cvt, r_ts in h_r_t.items():
            flag = True
            at_leat_one_flag = False
            for r_t in r_ts:
                rel, obj = r_t

                if rel.endswith(".from"):
                    rel = rel.rstrip(".from")
                    rel = rel + ".start_time"
                if rel.endswith(".to"):
                    rel = rel.rstrip(".to")
                    rel = rel + ".end_time"
                rel_surface = rel

                # object
                if obj in self.mid_mapping_dict:
                    obj_surface = self.mid_mapping_dict[obj]
                elif obj.startswith("m.") or obj.startswith('g.'):
                    # print("tail entity %s doesn't have name, we skip this triple." % obj)
                    continue
                else:
                    obj_surface = obj

                if rel_surface.lower() in constraint_response.lower():
                    at_leat_one_flag = True
                    if obj_surface.lower() not in constraint_response.lower():
                        flag = False
                        break
            if flag and at_leat_one_flag:
                valid_cvt_nodes.add(cvt)

        # 添加软约束条件，解析cvt结点的rel，若有两部分在response中则选中
        if len(valid_cvt_nodes) == 0:
            for cvt, r_ts in h_r_t.items():
                flag = True
                at_leat_one_flag = False
                for r_t in r_ts:
                    rel, obj = r_t

                    if rel.endswith(".from"):
                        rel = rel.rstrip(".from")
                        rel = rel + ".start_time"
                    if rel.endswith(".to"):
                        rel = rel.rstrip(".to")
                        rel = rel + ".end_time"
                    rel_surface = rel

                    # object
                    if obj in self.mid_mapping_dict:
                        obj_surface = self.mid_mapping_dict[obj]
                    elif obj.startswith("m.") or obj.startswith('g.'):
                        # print("tail entity %s doesn't have name, we skip this triple." % obj)
                        continue
                    else:
                        obj_surface = obj

                    rel_surface_list = rel_surface.split(".")
                    for rel in rel_surface_list:
                        if rel.lower() in constraint_response.lower():
                            at_leat_one_flag = True
                            if obj_surface.lower() not in constraint_response.lower():
                                flag = False
                                break
                            else:
                                flag = True
                    if flag and at_leat_one_flag:
                        valid_cvt_nodes.add(cvt)
                        break

        new_tris_per_hop = defaultdict(set)
        for hop in [0, 1]:
            triples = filtered_triples_per_hop[hop]
            for tri in triples:
                h, r, t = tri
                if hop == 0:
                    if t in valid_cvt_nodes:
                        new_tris_per_hop[hop].add(tuple(tri))
                elif hop == 1:
                    if h in valid_cvt_nodes:
                        new_tris_per_hop[hop].add(tuple(tri))
        return new_tris_per_hop

    def serialize_facts_one_hop(self, facts):
        if len(facts) > 0:
            h_r_t = defaultdict(lambda: defaultdict(set))
            visited_flag = {}
            for fact in facts:
                h, r, t = fact
                visited_flag[tuple(fact)] = False
                h_r_t[h][r].add(t)
            facts_str = []
            for tri in facts:
                if not visited_flag[tuple(tri)]:
                    h, r, t = tri
                    if self.is_cvt(t) and len(h_r_t[t]) == 0:
                        continue
                    if self.is_cvt(h):
                        # print("Qid:[%s] has single cvt head entities." % qid)
                        # logger.info(triples_per_hop)
                        continue
                    elif self.is_cvt(t):
                        one_hop_triples = h_r_t[t]
                        if len(one_hop_triples) > 0:
                            h_new = t
                            r_new = []
                            t_new = []
                            for key_r, value_ts in one_hop_triples.items():
                                for t_ in value_ts:
                                    visit_tri = (t, key_r, t_)
                                    if not visited_flag[visit_tri]:
                                        r_new.append(key_r)
                                        t_new.append(t_)
                                        visited_flag[visit_tri] = True
                            tri_new = (h, r_new, t_new)
                            if len(r_new) == len(t_new) > 0:
                                str_tri_list = self.convert_hyper_facts_to_text(tri_new)
                                if str_tri_list is not None:
                                    for st in str_tri_list:
                                        assert len(st) == 3
                                        if st not in facts_str:
                                            facts_str.append(st)
                    else:
                        st = self.convert_fact_to_text(tri)
                        if st is not None:
                            assert len(st) == 3
                            if st not in facts_str:
                                facts_str.append(st)
            facts_str = ["(" + ", ".join(fact) + ")" for fact in facts_str]
            serialized_facts = ""
            for fact in facts_str:
                serialized_facts_tmp = serialized_facts + fact + "; "
                if len(serialized_facts_tmp.split()) > self.max_serialization_tokens:
                    break
                else:
                    serialized_facts = serialized_facts_tmp
            # serialized_facts = "; ".join(facts_str)
            serialized_facts = serialized_facts.strip("; ")
        else:
            serialized_facts = ""
        return serialized_facts

    def is_end_v2(self, response, iterative_step):
        if "final" in response.lower() or iterative_step > 3:
            return True
        else:
            return False

    def reset_history(self):
        self.log = []
        self.selected_relations = []
        self.selected_sub_questions = []

    def get_tails_list(self, cur_ents):
        tails = self.SLM.get_tails_list(cur_ents)
        return tails

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
            for sample in tqdm(all_data, total=len(all_data)):
                # if sample["ID"] not in ["test_10943"]:
                #     continue
                try:
                    question = sample["Question"]
                    tpe_name = sample["TopicEntityName"]
                    tpe_id = sample['TopicEntityID']

                    prediction, chat_history, record = solver.forward_v2(question, tpe_name, tpe_id)
                    valid_count += 1
                except openai.error.InvalidRequestError as e:
                    print(e)
                    continue
                except Exception as e:
                    logging.exception(e)
                    continue

                chat = sample["ID"] + "\n" + "\n******\n".join(chat_history) + "\nAnswers: " + str(
                    sample['Answers']) + "\n------------------------------------------\n"
                fclog.write(chat)

                count += 1
                if count < 5:
                    print(sample['Answers'])
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
    parser.add_argument('--log_path', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--prompt_path')
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--bagging_type', default="llm", )
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--device', default=0, help='the gpu device')
    parser.add_argument('--topk', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--max_tokens', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--api_key', default="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", type=str)
    parser.add_argument('--filter_score', default=0.0, type=float, help='the minimal cosine similarity')
    parser.add_argument('--kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--rel2id_path', default=None, help='the sparse rel2id file')
    parser.add_argument('--ent2name_path', default=None, help='the sparse rel2id file')
    parser.add_argument('--max_triples_per_relation', default=40, type=int)
    parser.add_argument('--max_llm_input_tokens', default=3400, type=int)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')


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

    # used for interrupted scenario
    # with open(args.output_path, "r") as f:
    #     all_lines = f.readlines()
    #     all_lines = [json.loads(line.strip("\n")) for line in all_lines]
    #     already_id = [line['ID'] for line in all_lines]
    #     all_data = [data for data in all_data if data['ID'] not in already_id]
    #     print("There are %d test examples need to be processed." % len(all_data))

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