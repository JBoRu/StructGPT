import math
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
VERY_LARGT_NUM = 10**8
PATH_CUTOFF = 10**6
NODE_CUTOFF = 10**4


class KnowledgeGraphSparse(object):
    def __init__(self, triples_path: str, ent_type_path: str):
        self.triple = self._load_npy_file(triples_path)
        self.ent_type = self._load_npy_file(ent_type_path)
        self.bin_map = np.zeros_like(self.ent_type, dtype=np.int32)
        self.E = self.triple.shape[0]
        self.head2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 0], np.arange(self.E)))).astype('bool')
        self.rel2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 1], np.arange(self.E)))).astype('bool')
        self.tail2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 2], np.arange(self.E)))).astype('bool')
        self.max_head = max(self.triple[:, 0])
        self.max_tail = max(self.triple[:, 2])
        self.last_tails = set()

    @staticmethod
    def _load_npy_file(filename):
        return np.load(filename)

    def _fetch_forward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.max_head)
        indices = self.head2fact[seed_set].indices
        return self.triple[indices]

    def _fetch_backward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.max_tail)
        indices = self.tail2fact[seed_set].indices
        return self.triple[indices]

    def filter_cvt_nodes(self, seed_ary, CVT_TYPE=3):
        seed_type = self.ent_type[seed_ary]
        return seed_ary[seed_type == CVT_TYPE]

    def get_facts_1hop(self, seed_set, id2rel, rel2id, max_triples_per_relation, gold_relations):
        filtered_triples = defaultdict(list)
        filtered_tails = []

        triples = self._fetch_forward_triple(seed_set)
        if len(triples) == 0:
            print("No triples")
            return filtered_triples, filtered_tails

        cur_heads = set()
        cur_heads.update(seed_set)

        candidate_rels = set(triples[:, 1].tolist())
        candidate_rels_str = [id2rel[rel] for rel in candidate_rels]

        if len(candidate_rels_str) == 0:
            print("No candidate_rels_str")
            return filtered_triples, filtered_tails

        if gold_relations is None:
            filtered_rels_str = set(candidate_rels_str)
        else:
            filtered_rels_str = set(candidate_rels_str) & set(gold_relations)
            assert len(filtered_rels_str) == len(set(gold_relations))

        for rel_str in filtered_rels_str:
            rel_indices = (triples[:, 1] == rel2id[rel_str])
            triples_for_rel = triples[rel_indices]
            # if len(triples_for_rel) > max_triples_per_relation:
            #     continue
            for triple in triples_for_rel.tolist():
                if triple[2] not in self.last_tails:
                    filtered_triples[0].append(triple)
                    filtered_tails.append(triple[2])

        cvt_tails = [tail for tail in filtered_tails if self.ent_type[tail] == 3]
        filtered_tails = list(set(filtered_tails) - set(cvt_tails))
        if len(cvt_tails) > 0:
            triples = self._fetch_forward_triple(cvt_tails)
            if len(triples) > 0:
                cur_heads.update(cvt_tails)

                cur_invalid_rels = set()
                candidate_rels = set(triples[:, 1].tolist())
                candidate_rels_str = [id2rel[rel] for rel in candidate_rels if rel not in cur_invalid_rels]
                filtered_rels_str = candidate_rels_str
                for rel_str in filtered_rels_str:
                    rel_indices = (triples[:, 1] == rel2id[rel_str])
                    triples_for_rel = triples[rel_indices].tolist()
                    for triple in triples_for_rel:
                        if triple[2] not in seed_set:
                            if self.ent_type[triple[2]] != 3:
                                filtered_triples[1].append(triple)
                                filtered_tails.append(triple[2])

        self.last_tails = deepcopy(cur_heads)
        filtered_tails = list(set(filtered_tails))
        return filtered_triples, filtered_tails

    def get_rels_1hop(self, seed_set, id2rel):
        triples = self._fetch_forward_triple(seed_set)
        if len(triples) == 0:
            return []

        cur_heads = set()
        cur_heads.update(seed_set)

        cur_invalid_rels = set()
        for tail in self.last_tails:
            invalid_triples_indices = (triples[:, 2] == tail)
            invalid_triples = triples[invalid_triples_indices]
            invalid_rels = set(invalid_triples[:, 1])
            cur_invalid_rels.update(invalid_rels)

        candidate_rels = set(triples[:, 1].tolist())
        candidate_rels_str = [id2rel[rel] for rel in candidate_rels if rel not in cur_invalid_rels]

        return candidate_rels_str

    def get_filtered_rels(self, question, cur_relations, tokenizer, model, topk, filter_score):
        scored_rel_list, filtered_rel_scored_list = self.score_relations(question, cur_relations, tokenizer, model, filter_score)
        # 过滤关系和得分
        ordered_rels_scored = sorted(filtered_rel_scored_list, key=lambda x: x[1], reverse=True)
        # 过滤方法为topk和最少路径filter_method == "topk":
        reserved_rels = ordered_rels_scored[:topk]
        reserved_rels = [rel_score[0] for rel_score in reserved_rels]
        return reserved_rels


