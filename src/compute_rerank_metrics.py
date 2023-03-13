import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json

def extract_pids(data):
    all_pids = []
    for i in tqdm(range(len(data))):
        row = data[i]

        if type(row['output']) == str:
            row_list = eval(row['output'])
        else:
            row_list = row['output']
        pids = []
        for x in row_list[0]['provenance']:
            pid = int(x['wikipedia_id'])
            pids.append(pid)
        all_pids.append(pids)
    return all_pids

def extract_positive_pids(data):
    all_positive_pids = []
    for i in tqdm(range(len(data))):
        pos_pid = int(eval(data[i])[0])
        all_positive_pids.append(pos_pid)
    return all_positive_pids


def compute_recall_k(positive_pids, pred_pids, k=1):
    flags = [0] * len(positive_pids)

    for i in range(len(positive_pids)):
        pos_pid = positive_pids[i]
        preds = pred_pids[i][:k]

        if pos_pid in preds:
            flags[i] = 1

    return sum(flags)/len(flags)

def load_pred_dataset(file_path):
    json_list = {"output":[]}
    with open(file_path, "r") as f:
        for line in f:
            temp = json.loads(line)
            json_list["output"].append(temp['output'])
    return Dataset.from_dict(json_list)
    

def compute_rerank_mertics(args):
    gt_data = load_dataset("json", data_files=args.ground_truth_file)['train']
    pred_data = load_pred_dataset(args.predictions_file)
    gt_positive_pids = extract_positive_pids(gt_data['positive_pids'])
    pred_pids = extract_pids(pred_data)

    for k in [1, 5, 10, 20]:
        print(f"Recall@{k} = {compute_recall_k(gt_positive_pids, pred_pids, k)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gf', '--ground_truth_file', type=str)
    parser.add_argument('-pf', '--predictions_file', type=str)
    args = parser.parse_args()

    compute_rerank_mertics(args)