import json
import numpy as np
import argparse


def evaluate(ori_path, inp_path, error_cases_output, write_flag):
    with open(ori_path, "r") as f:
        all_data = json.loads(f.read())
        print("Totally %d test data" % len(all_data))

    pred_data = []
    with open(inp_path, "r") as f:
        lines = f.readlines()
        datas = [json.loads(line) for line in lines]
        pred_data.extend(datas)

    all_pred_data = {pred['id']: pred for pred in pred_data}
    print("Totally %d prediction data" % len(pred_data))  # evaluate_is_right
    avg_acc = []
    bad_cases = []
    error_count = 0
    max_count = 0
    right_count = 0
    for data in all_data:
        if data["id"] in all_pred_data:
            data = all_pred_data[data["id"]]
            question = data['statement']
            pred = data['Prediction'].lower()

            if 'yes' in pred and 'no' in pred:
                pred = 'unknown'
            elif 'yes' in pred:
                pred = 'entailed'
            elif 'no' in pred:
                pred = 'refuted'
            else:
                pred = 'unknown'

            answers = data['seq_out'].lower()
            if pred.strip() == answers.strip():
                avg_acc.append(1)
                right_count += 1

            else:
                error_count += 1
                avg_acc.append(0)
                print("ID: %s Ques: %s" % (data["id"], question))
                print("Pred: ", pred)
                print("Ans: ", answers)
                print("------------------------------------------------------------------------")
                bad_cases.append(data)

        else:
            avg_acc.append(0)
            print("ID: %s can't be predicted" % (data["id"]))
            bad_cases.append(data)
            error_count += 1
            max_count += 1

    acc = np.mean(avg_acc)
    print("Acc: %.4f" % (acc))
    if write_flag:
        with open(error_cases_output, "w") as f:
            for bc in bad_cases:
                f.write(json.dumps(bc) + "\n")
    print("Totally %d bad cases need further solved." % len(bad_cases))
    print("Right count: %d, Error count: %d(Max len count: %d)" % (right_count, error_count, max_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./data/tabfact/tab_fact_test.json")
    parser.add_argument('--inp_path', type=str, default="./outputs/tabfact/tabfact_test_output.jsonl")
    parser.add_argument('--error_cases_output', type=str,
                        default='./outputs/tabfact/bad_cases.jsonl')
    parser.add_argument('--write_flag', action="store_true")
    args = parser.parse_args()
    evaluate(args.ori_path, args.inp_path, args.error_cases_output, args.write_flag)
