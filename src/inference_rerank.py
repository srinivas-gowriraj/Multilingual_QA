import torch
import os
import json
from tqdm import tqdm
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRerankPreprocessor
from typing import Union
import argparse
from datasets import load_dataset
from config import hparams

hp = hparams()

class myDocumentGroundedDialogRerankPipeline(DocumentGroundedDialogRerankPipeline):
    def __init__(self,
                 model: Union[DocumentGroundedDialogRerankModel, str],
                 preprocessor: DocumentGroundedDialogRerankPreprocessor = None,
                 config_file: str = None,
                 device: str = 'cuda',
                 auto_collate=True,
                 seed: int = 88,
                 **kwarg):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            seed=seed,
            **kwarg
        )

    def save(self, addr):
        file_out = open(addr, 'w')
        for every_dict in self.guess:
            file_out.write(json.dumps(every_dict) + '\n')


def process_retr_result_file(queries_file_path, retrieved_passages_file_path):
    queries = open(queries_file_path, 'r')
    all_queries = []
    for every_query in queries:
        all_queries.append(json.loads(every_query))
    passage_to_id = {}
    ptr = -1
    for file_name in ['fr', 'vi']:
        with open(os.path.join(hp.all_passages_dir, f'{file_name}.json')) as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage] = str(ptr)

    retrieved_passages_file = open(retrieved_passages_file_path, 'r')
    retrieval_result = json.load(retrieved_passages_file)['outputs']
    input_list = []
    passages_list = []
    ids_list = []
    output_list = []
    positive_pids_list = []
    ptr = -1
    for x in tqdm(all_queries):
        ptr += 1
        now_id = str(ptr)
        now_input = x
        now_wikipedia = []
        now_passages = []
        all_candidates = retrieval_result[ptr]
        for every_passage in all_candidates:
            get_pid = passage_to_id[every_passage]
            now_wikipedia.append({'wikipedia_id': str(get_pid)})
            now_passages.append({"pid": str(get_pid), "title": "", "text": every_passage})
        now_output = [{'answer': '', 'provenance': now_wikipedia}]
        input_list.append(now_input['query'])
        passages_list.append(str(now_passages))
        ids_list.append(now_id)
        output_list.append(str(now_output))
        positive_pids_list.append(str([]))
    evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                        'positive_pids': positive_pids_list}
    return evaluate_dataset

def convert_int_to_str(example):
    example["id"] = str(example["id"])
    return example

def main(cli_args):
    model_dir = './output'
    model_configuration = {
        "framework": "pytorch",
        "task": "document-grounded-dialog-rerank",
        "model": {
            "type": "doc2bot"
        },
        "pipeline": {
            "type": "document-grounded-dialog-rerank"
        },
        "preprocessor": {
            "type": "document-grounded-dialog-rerank"
        }
    }
    file_out = open(f'{model_dir}/configuration.json', 'w')
    json.dump(model_configuration, file_out, indent=4)
    file_out.close()
    args = {
        'output': './',
        # 'max_batch_size': 16,
        'max_batch_size': 64,
        'exclude_instances': '',
        'include_passages': False,
        'do_lower_case': True,
        'max_seq_length': 512,
        'query_length': 195,
        'tokenizer_resize': True,
        'model_resize': True,
        'kilt_data': True
    }
    model = Model.from_pretrained(model_dir, **args)
    mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
        model.model_dir, **args)
    pipeline_ins = myDocumentGroundedDialogRerankPipeline(
        model=model, preprocessor=mypreprocessor, **args)

    if cli_args.preprocess:
        evaluate_dataset = process_retr_result_file(cli_args.queries_file_path, cli_args.retrieved_passages_file_path)
    else:
        evaluate_dataset = load_dataset('json', data_files=cli_args.eval_dataset_file_path)['train'].map(convert_int_to_str)
        evaluate_dataset = {key: evaluate_dataset[key] for key in evaluate_dataset.features}
        
    pipeline_ins(evaluate_dataset)
    pipeline_ins.save(f'./rerank_output.jsonl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rpfp", "--retrieved_passages_file_path", default=os.path.join(hp.root_dir, 'src', 'DAMO_ConvAI', 'nlp_convai_retrieval_pretrain', 'evaluate_result.json'), type=str)
    parser.add_argument("-qfp", "--queries_file_path", default=os.path.join(hp.data_dir, 'input.jsonl'), type=str)
    parser.add_argument("-edfp", "--eval_dataset_file_path", help="File path to json file containing processed eval data", type=str)
    parser.add_argument('-pr', "--preprocess", action="store_true")

    args = parser.parse_args()
    main(args)
