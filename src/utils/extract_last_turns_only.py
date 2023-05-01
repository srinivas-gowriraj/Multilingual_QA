from datasets import load_dataset
import json, os
import argparse

def extract_last_turn(example):
    query = example['query']
    query.replace("\"", "\\\"")
    agent_index = query.index("<agent>") if "<agent>" in query else -1
    user_index = query.index("<user>") if "<user>" in query else -1
    stop_index = agent_index if agent_index < user_index else user_index
    if stop_index != -1:
        example['query'] = query[:stop_index]
    return example

def process_dataset(args):
    ds = load_dataset('json', data_files=args.input_file_path)["train"]
    ds = ds.map(extract_last_turn)
    # with open(args.output_file_path, "w") as f:
    #     for i in range(0, len(ds)):
    #         text = f"""{{"query":"{ds[i]['query']}", "positive":"{ds[i]['positive']}", "negative":"{ds[i]['negative']}" }}\n"""
    #         f.write(text)
    #         f.flush()
    ds.to_json(args.output_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ifp', '--input_file_path', type=str, default="/usr0/home/sohamdit/Multilingual_QA/src/data/gpt_splits/EnDoc2BotRetrieval_train.json")
    parser.add_argument('-ofp', '--output_file_path', type=str, default="/usr0/home/sohamdit/Multilingual_QA/src/data/gpt_splits/EnDoc2BotRetrieval_2k_last_turn_train.json")
    args = parser.parse_args()
    process_dataset(args)