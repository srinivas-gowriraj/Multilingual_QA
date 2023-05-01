import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from datasets import load_dataset
import argparse
from config import hparams
from utils.labse_retriever import DocumentGroundedDialogRetrievalTrainerLabse
from utils.huggingface_retriever import DocumentGroundedDialogRetrievalTrainerHF

hp = hparams()
'''with open('data/dev.json') as f_in:
    with open('data/input.jsonl', 'w') as f_out:
        for line in f_in.readlines():
            sample = json.loads(line)
            sample['positive'] = ''
            sample['negative'] = ''
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open('data/input.jsonl') as f:
    eval_dataset = [json.loads(line) for line in f.readlines()]'''


def main(args):
    temp_datasets = []
    passage_languages = []
    lang_data_paths = hp.lang_data_paths

    if "french" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['french']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('fr')
    if "french_gpt" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['french_gpt']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('fr')
    if "french_2k_last_turn" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['french_2k_last_turn']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('fr')
    if "vietnamese" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['vietnamese']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('vi')
    if "vietnamese_gpt" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['vietnamese_gpt']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('vi')
    if "vietnamese_2k_last_turn" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['vietnamese_2k_last_turn']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('vi')
    
    eval_dataset = [x for dataset in temp_datasets for x in dataset]
    if args.domain is not None:
        eval_dataset_fr = []
        eval_dataset_vi = []
        if "french" in args.languages:
            eval_dataset_fr = [i for i in eval_dataset if i["positive"].split(
                "//")[-1] == (" fr" + "-" + args.domain)]
        if "vietnamese" in args.languages:
            eval_dataset_vi = [i for i in eval_dataset if i["positive"].split(
                "//")[-1] == (" vi" + "-" + args.domain)]
        eval_dataset = eval_dataset_fr + eval_dataset_vi

    if args.leaderboard_submission:
        with open(args.leaderboard_input_file) as f_in:
            with open('input.jsonl', 'w') as f_out:
                for line in f_in.readlines():
                    sample = json.loads(line)
                    sample['positive'] = ''
                    sample['negative'] = ''
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
        with open('input.jsonl') as f:
            eval_dataset = [json.loads(line) for line in f.readlines()]

    all_passages = []
    for file_name in set(passage_languages):
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)

    if args.model_type == "xlmr":
        # cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
        # trainer = DocumentGroundedDialogRetrievalTrainer(
        #     model=cache_path,
        #     train_dataset=None,
        #     eval_dataset=eval_dataset,
        #     all_passages=all_passages)
        cache_path = './DAMO_ConvAI/nlp_convai_retrieval_pretrain'
        trainer = DocumentGroundedDialogRetrievalTrainer(
            model=cache_path,
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages
        )

        if args.model_checkpoint is None:
            trainer.evaluate(
                checkpoint_path=os.path.join(trainer.model.model_dir,
                                             'finetuned_model.bin'))
        else:
            trainer.evaluate(
                checkpoint_path=args.model_checkpoint)
    elif args.model_type == "labse":
        output_file_path = args.model_checkpoint
        if args.model_checkpoint is None:
            raise Exception("Model checkpoint cannot be empty for LaBSE")
        trainer = DocumentGroundedDialogRetrievalTrainerLabse(
            model_save_path=output_file_path,
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "roberta_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="roberta-base",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "xlmr_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="xlm-roberta-base",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "bert_fr_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="dbmdz/bert-base-french-europeana-cased",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "bert_vi_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="trituenhantaoio/bert-base-vietnamese-uncased",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "mbert_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="bert-base-multilingual-cased",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)
    elif args.model_type == "bert_chinese_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.model_checkpoint,
            hf_checkpoint="bert-base-chinese",
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--model_type', type=str, default='xlmr',
                        choices=['xlmr', 'labse', 'xlmr_hf', 'roberta_hf', 'bert_fr_hf', 'bert_vi_hf', 'mbert_hf', 'bert_chinese_hf'])
    parser.add_argument("-l", '--languages', nargs='+',
                        default=["french", "vietnamese"])
    parser.add_argument('-mc', '--model_checkpoint',
                        type=str, required=False, default=None)
    parser.add_argument('-ls', '--leaderboard_submission', action="store_true")
    parser.add_argument('-lb', '--leaderboard_input_file', type=str,
                        required=False, default=hp.leaderboard_input_file)
    parser.add_argument('-d', '--domain', type=str, required=False, default=None,
                        help="Specify the domain for Vietnamese and French data")
    args = parser.parse_args()
    main(args)
