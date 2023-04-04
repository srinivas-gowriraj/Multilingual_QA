import os, shutil
import json
import argparse
import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from modelscope.utils.constant import DownloadMode
from datasets import load_dataset

from config import hparams
from utils.labse_retriever import DocumentGroundedDialogRetrievalTrainerLabse
from utils.huggingface_retriever import DocumentGroundedDialogRetrievalTrainerHF

hp = hparams()

def main(args):
    if ".bin" != args.output_file_path[-4:]:
        raise Exception("Provide output_file_path with the desired name and '.bin' extension")

    all_passages = []
    train_dataset = []
    val_dataset = []
    #lang_data_paths = hp['lang_data_paths']
    lang_data_paths = hp.lang_data_paths
    for language in args.languages:
        #retr_train_fp = f"{lang_data_paths[language]["retrieval"]["path"]}_train.json"
        #retr_val_fp = f"{lang_data_paths[language]["retrieval"]["path"]}_val.json"
        retr_train_fp = f"{lang_data_paths[language]['stages']['retrieval']['path']}_train.json"
        retr_val_fp = f"{lang_data_paths[language]['stages']['retrieval']['path']}_val.json"
        train_dataset.append(load_dataset('json', data_files=retr_train_fp)["train"])
        val_dataset.append(load_dataset('json', data_files=retr_val_fp)["train"])
        with open(lang_data_paths[language]["passage_path"], "r") as f:
            all_passages += json.load(f)

    train_dataset = [x for dataset in train_dataset for x in dataset]
    val_dataset = [y for dataset in val_dataset for y in dataset]
    if args.model_type == "xlmr":
        cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
        trainer = DocumentGroundedDialogRetrievalTrainer(
            model=cache_path,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            all_passages=all_passages)
    elif args.model_type == "labse":
        trainer = DocumentGroundedDialogRetrievalTrainerLabse(
            model_save_path=args.output_file_path,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            all_passages=all_passages)
    elif args.model_type == "roberta_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.output_file_path,
            hf_checkpoint="roberta-base",
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            all_passages=all_passages)
    elif args.model_type == "xlmr_hf":
        trainer = DocumentGroundedDialogRetrievalTrainerHF(
            model_save_path=args.output_file_path,
            hf_checkpoint="xlm-roberta-base",
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            all_passages=all_passages)

    if args.model_checkpoint is not None:
        state_dict = torch.load(args.model_checkpoint)
        if args.model_type == "xlmr":
            trainer.model.model.load_state_dict(state_dict)
        else:
            trainer.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {args.model_checkpoint}. Will continue training.")

    trainer.train(
        # batch_size=128,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        total_epoches=args.num_epochs,
    )

    if args.model_type == "xlmr":
        output_dir = os.path.dirname(args.output_file_path)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(os.path.join(trainer.model.model_dir,
                                        'finetuned_model.bin'), output_dir)
        os.rename(f"{output_dir}/finetuned_model.bin", args.output_file_path)
        print(f"Saved trained retrieval model weights after training in: {args.output_file_path}")

        print(f"Starting evaluation using weights saved in above...")
        
    trainer.evaluate(
        checkpoint_path=args.output_file_path)
    print(f"Evaluation of retrieval model complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--model_type', type=str, default='xlmr', choices=['xlmr', 'labse', 'xlmr_hf', 'roberta_hf'])
    parser.add_argument("-l", '--languages', nargs='+', default=hp.available_languages)
    parser.add_argument("-ofp", '--output_file_path', type=str, required=True, help="File path where you want to save the retrieval model weights, with '.bin' extension.")
    parser.add_argument('-mc', '--model_checkpoint', type=str, required=False, default=None, help="input model checkpoint where you want to load weights and continue training from")
    parser.add_argument('-as', '--accumulation_steps', type=int, required=False, default=8, help="Number of gradient accumulation steps")
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=16, help="Model batch size")
    parser.add_argument('-ne', '--num_epochs', type=int, required=False, default=50, help="Number of epochs to train model")
    args = parser.parse_args()
    main(args)


# ret -> top 20 pids
# reranker -> cross-encoder query, top 20 (contains gold R@20=1), (q+pi) cls repr, cls - linear - score, for all 20 pairs, loss = exp(q+p_gold)/sum(exp(q+pi)) 
#   how to chose: p- -> at entire index level 
    # train better reranker using french, negatives pids are non-gold retrieved pids

# not train anything, just evaluate using their checkpoint without training
# 
