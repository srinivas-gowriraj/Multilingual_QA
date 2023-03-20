# ONLY RUN THIS FILE IF YOU HAVE ACCESS TO THE PASSAGES OF ALL LANGUAGES
import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset

from config import hparams
hp = hparams()
def main(args):
    # 4. generate rerank data
    passage_to_id = {}
    id_to_passage = {}
    ptr = -1

    for file_name in hp.available_languages_short_names:
        with open(os.path.join(hp.all_passages_dir, f'{file_name}.json')) as f:
            all_passages = json.load(f)
            for each_passage in tqdm(all_passages):
                ptr += 1
                passage_to_id[each_passage] = str(ptr)
                passage_to_id[str(ptr)] = each_passage
    if args.save_id_to_passage:
        id_to_passage_path = os.path.join(hp.all_passages_dir, "id_to_passage.json")
        with open(id_to_passage_path, "w") as f:
            f.write(json.dumps(id_to_passage, indent=4))
        print(f"Saved id_to_passage dict to path: {id_to_passage_path}")

    for lang in args.languages:
        print(f"Starting generating reranker data for {lang}...")
        original_dialogue_data = load_dataset('json', data_files=os.path.join(f"{hp.lang_data_paths[lang]['default_data_path']}.json"))['train']
        input_list = []
        passages_list = []
        ids_list = []
        output_list = []
        positive_pids_list = []
        ptr = -1
        for i in tqdm(range(len(original_dialogue_data))):
            x = original_dialogue_data[i]
            ptr += 1
            now_id = str(ptr)
            now_input = x
            now_wikipedia = []
            now_passages = []
            all_candidates = eval(x["passages"])
            pos_pids = []
            for each_passage in all_candidates:
                get_pid = passage_to_id[each_passage]
                pos_pids.append(str(get_pid))
                now_wikipedia.append({'wikipedia_id': str(get_pid)})
                now_passages.append({"pid": str(get_pid), "title": "", "text": each_passage})
            now_output = [{'answer': '', 'provenance': now_wikipedia}]
            input_list.append(now_input['query'])
            passages_list.append(str(now_passages))
            ids_list.append(now_id)
            output_list.append(str(now_output))
            positive_pids_list.append(str(pos_pids))
        reranker_dataset = Dataset.from_dict({'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                            'positive_pids': positive_pids_list})
        reranker_dataset = reranker_dataset.train_test_split(test_size=hp.test_size, seed=hp.seed)
        rerank_train_path = f"{hp.lang_data_paths[lang]['rerank']['path']}_train.json"
        reranker_dataset["train"].to_json(rerank_train_path)
        rerank_val_path = f"{hp.lang_data_paths[lang]['rerank']['path']}_val.json"
        reranker_dataset["test"].to_json(rerank_val_path)
        print(f"Saved reranker train and val splits for: {lang} at {rerank_train_path} and {rerank_val_path}.")
        breakpoint()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--languages', nargs='+', choices=["english", "chinese"], default=["english", "chinese"])
    parser.add_argument("--save_id_to_passage", action="store_true")
    args = parser.parse_args()
    main(args)