import argparse

import openai
import json
import os
import numpy as np
from collections import defaultdict
import pickle

import json
import numpy as np

def evaluate(ori_path, pred_path, error_cases_output, write_flag):
    with open(ori_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(line) for line in all_data]

    with open(pred_path, "r", encoding="UTF-8") as f:
        all_lines = f.readlines()
        all_pred_data = []
        for idx, line in enumerate(all_lines):
            line = line.replace("\x00", "").strip("\n")
            all_pred_data.append(json.loads(line))
        all_pred_data = {pred['ID']: pred for pred in all_pred_data}
    print("Load %d prediction" % len(all_pred_data))

    max_len_count = len(all_data) - len(all_pred_data)
    print("Totally %d prediction / %d all data" % (len(all_pred_data), len(all_data)))

    avg_hits1 = []
    bad_cases = []
    right_cases_id = []
    bad_cases_id = []
    right_count = 0
    bad_count = 0
    need_cvt_count = 0
    for data in all_data:
        if data["ID"] in all_pred_data:
            data = all_pred_data[data["ID"]]
            question = data['Question']
            pred = data['Prediction'].lower()

            answers = data['Answers']
            aliases = data['Aliases']
            hit_flag = []
            recall_flag = []
            for ans in answers:
                ans = ans.lower()
                if ans in pred:
                    hit_flag.append(1)
                    recall_flag.append(1)
                else:
                    hit_flag.append(0)
                    recall_flag.append(0)
            for alia in aliases:
                alia = alia.lower()
                if alia in pred:
                    hit_flag.append(1)
                else:
                    hit_flag.append(0)

            if len(hit_flag) == 0:
                # print("ID:%s doesn't have any gold answers." % data['ID'])
                continue

            if any(hit_flag):
                avg_hits1.append(1)
                right_count += 1
                # other_count += 1
                right_cases_id.append(data['ID'])
            else:
                avg_hits1.append(0)
                bad_count += 1
                # other_count += 1
                # if "max length" in pred:
                #     need_cvt_count += 1
                # else:
                #     other_count += 1
                print(data["ID"])
                print("ID: %s Ques: %s" % (data["ID"], question))
                print("Pred: ", pred)
                print("Ans: ", answers)
                print("------------------------------------------------------------------------")
                bad_cases.append(data)
                bad_cases_id.append(data["ID"])

        else:
            avg_hits1.append(0)
            print("ID: %s can't be predicted" % (data["ID"]))
            bad_cases.append(data)
            bad_cases_id.append(data["ID"])

    hits1 = np.mean(avg_hits1)
    print("Hits@1: %.4f" % (hits1))
    if write_flag:
        with open(error_cases_output, "w") as f:
            for bc in bad_cases:
                f.write(json.dumps(bc) + "\n")
    print("Totally %d bad cases need further solved." % len(bad_cases))
    print("Right:%d, Wrong:%d, Max_len:%d" % (right_count, bad_count, max_len_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str)
    parser.add_argument('--inp_path', type=str)
    parser.add_argument('--error_cases_output', action="store_true")
    parser.add_argument('--write_flag', action="store_true")
    args = parser.parse_args()
    evaluate(args.ori_path, args.inp_path, args.error_cases_output, args.write_flag)
