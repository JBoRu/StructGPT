import argparse
import json
from collections import defaultdict
import numpy as np


def evaluate_example(_predict_str: str, _ground_str: list, target_delimiter=', '):
    _predict_spans = _predict_str.split(target_delimiter)
    _predict_spans = [x.lower().strip().strip('.').strip("'").strip('"').strip() for x in _predict_spans]
    for i in range(len(_predict_spans)):
        if _predict_spans[i].endswith('.0'):
            _predict_spans[i] = _predict_spans[i][:-2]

        if _predict_spans[i].replace(',', '').isnumeric():
            _predict_spans[i] = _predict_spans[i].replace(',', '')
    # _ground_spans = _ground_str.split(target_delimiter)
    _ground_spans = [x.lower().strip().strip('.').strip("'").strip('"').strip() for x in _ground_str]
    for i in range(len(_ground_spans)):
        if _ground_spans[i].endswith('.0'):
            _ground_spans[i] = _ground_spans[i][:-2]

        if _ground_spans[i].replace(',', '').isnumeric():
            _ground_spans[i] = _ground_spans[i].replace(',', '')
    _predict_values = defaultdict(lambda: 0)
    _ground_values = defaultdict(lambda: 0)
    for span in _predict_spans:
        try:
            _predict_values[float(span)] += 1
        except ValueError:
            _predict_values[span.strip()] += 1
    for span in _ground_spans:
        try:
            _ground_values[float(span)] += 1
        except ValueError:
            _ground_values[span.strip()] += 1
    _is_correct = _predict_values == _ground_values
    return _is_correct


def evaluate(ori_path, inp_path, error_cases_output, write_flag):
    with open(ori_path, "r") as f:
        all_data = json.loads(f.read())
        print("Totally %d test data" % len(all_data))

    pred_data = []
    with open(inp_path, "r") as f:
        lines = f.readlines()
        datas = [json.loads(line) for line in lines]
        pred_data.extend(datas)

    all_pred_data = {pred['question']: pred for pred in pred_data}
    print("Totally %d prediction data" % len(pred_data))  # evaluate_is_right
    avg_deno_acc = []
    bad_cases = []
    error_count = 0
    max_count = 0
    right_count = 0
    for data in all_data:
        if data["question"] in all_pred_data:
            data = all_pred_data[data["question"]]
            pred = data['Prediction'].lower()

            if "answers: " in pred:
                pred = pred.split("answers: ")[1].strip()
            elif ":" in pred:
                pred = pred.split(":")[1].strip()
            else:
                pred = pred

            answers = data['answer_text']
            answers = [ans if not ans.endswith(".0") else ans.replace(".0", "") for ans in answers]

            if evaluate_example(pred, answers):
                avg_deno_acc.append(1)
                right_count += 1
            else:
                error_count += 1
                avg_deno_acc.append(0)
                # print("ID: %s Ques: %s" % (data["id"], question))
                # print("Pred: ", pred)
                # print("Ans: ", answers)
                # print("------------------------------------------------------------------------")
                bad_cases.append(data)
        else:
            avg_deno_acc.append(0)
            print("ID: %s can't be predicted" % (data["id"]))
            bad_cases.append(data)
            error_count += 1
            max_count += 1

    acc = np.mean(avg_deno_acc)
    print("Denotation Acc: %.4f" % (acc))
    if write_flag:
        with open(error_cases_output, "w") as f:
            for bc in bad_cases:
                f.write(json.dumps(bc) + "\n")
    print("Totally %d bad cases need further solved." % len(bad_cases))
    print("Right count: %d, Error count: %d(Max len count: %d)" % (right_count, error_count, max_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./data/wikisql/wikisql_test.json")
    parser.add_argument('--inp_path', type=str, default="./outputs/wikisql/output_wo_icl_v1.jsonl")
    parser.add_argument('--error_cases_output', type=str,
                        default='./outputs/wikisql/bad_cases.jsonl')
    parser.add_argument('--write_flag', action="store_true")
    args = parser.parse_args()
    evaluate(args.ori_path, args.inp_path, args.error_cases_output, args.write_flag)
