from collections import defaultdict
from copy import deepcopy

import logging
import numpy as np

from KnowledgeBase.KG_sparse_api import KnowledgeGraphSparse
import pickle

END_OF_HOP = "end.hop"
SEP = "[SEP]"


class KnowledgeGraph(object):
    def __init__(self, sparse_triples_path, sparse_ent_type_path, ent2id_path, rel2id_path):
        triples_path, ent_type_path = sparse_triples_path, sparse_ent_type_path
        print("The sparse KG instantiate via int triples from the %s" % (triples_path))
        self.sparse_kg = KnowledgeGraphSparse(triples_path=triples_path, ent_type_path=ent_type_path)
        self.ent2id = self._load_pickle_file(ent2id_path)
        self.id2ent = self._reverse_dict(self.ent2id)
        self.rel2id = self._load_pickle_file(rel2id_path)
        self.id2rel = self._reverse_dict(self.rel2id)
        print("The sparse KG instantiate over, all triples: %d, max head id: %d." % (
            self.sparse_kg.E, self.sparse_kg.max_head))

    @staticmethod
    def _load_pickle_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _reverse_dict(ori_dict):
        reversed_dict = {v: k for k, v in ori_dict.items()}
        return reversed_dict

    def get_facts_1hop(self, seeds, max_triples_per_relation, first_flag, gold_relations):
        if first_flag:
            seeds_id = []
            for seed in seeds:
                try:
                    seed_id = self.ent2id[seed]
                    seeds_id.append(seed_id)
                except Exception as e:
                    logging.exception(e)
                    print("Entity string: %s not in ent2id dict" % seed)
                    continue
        else:
            seeds_id = seeds
        if len(seeds_id) == 0:
            return defaultdict(list), []
        triples_per_hop, tails = self.sparse_kg.get_facts_1hop(seeds_id, self.id2rel, self.rel2id, max_triples_per_relation, gold_relations)
        triples_per_hop = {hop: [[self.id2ent[triple[0]], self.id2rel[triple[1]], self.id2ent[triple[2]]] for triple in triples]
                   for hop, triples in triples_per_hop.items()}
        return triples_per_hop, tails

    def get_rels_1hop(self, seeds, first_flag):
        if first_flag:
            seeds_id = []
            for seed in seeds:
                try:
                    seed_id = self.ent2id[seed]
                    seeds_id.append(seed_id)
                except Exception as e:
                    logging.exception(e)
                    print("Entity string: %s not in ent2id dict" % seed)
                    continue
        else:
            seeds_id = seeds
        if len(seeds_id) == 0:
            return []
        can_rels = self.sparse_kg.get_rels_1hop(seeds_id, self.id2rel)
        return can_rels
