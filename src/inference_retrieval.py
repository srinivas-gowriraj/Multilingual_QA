import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from datasets import load_dataset
import argparse
from config import hparams
from utils.labse_retriever import DocumentGroundedDialogRetrievalTrainerLabse
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
    if "french" in args.languages:
        temp_datasets.append(load_dataset('json', data_files='./data/splits/FrDoc2BotRetrieval_val.json')['train'])
        passage_languages.append('fr')
    if "vietnamese" in args.languages:
        temp_datasets.append(load_dataset('json', data_files='./data/splits/ViDoc2BotRetrieval_val.json')['train'])
        passage_languages.append('vi')
        
    eval_dataset = [x for dataset in temp_datasets for x in dataset]
        
    if args.leaderboard_submission:
        with open(args.leaderboard_file) as f_in:
            with open('input.jsonl', 'w') as f_out:
                for line in f_in.readlines():
                    sample = json.loads(line)
                    sample['positive'] = ''
                    sample['negative'] = ''
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
        with open('input.jsonl') as f:
            eval_dataset = [json.loads(line) for line in f.readlines()]

    
    all_passages = []
    for file_name in passage_languages:
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)
            
    
    if args.model_type == "xlmr":
        # cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
        # trainer = DocumentGroundedDialogRetrievalTrainer(
        #     model=cache_path,
        #     train_dataset=train_dataset,
        #     eval_dataset=val_dataset,
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
        trainer.evaluate()
                            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--model_type', type=str, default='xlmr', choices=['xlmr', 'labse'])
    parser.add_argument("-l", '--languages', nargs='+', default=["french", "vietnamese"])
    parser.add_argument('-mc', '--model_checkpoint', type=str, required=False, default=None)
    parser.add_argument('-ls', '--leaderboard_submission', type=bool, required=False, default=False)
    parser.add_argument('-lb', '--leaderboard_file', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
