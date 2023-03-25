import torch 
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

import os
import json
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from datasets import load_dataset
import pandas as pd


def main(args):
    fr_train_dataset = load_dataset('json', data_files="data/splits/FrDoc2BotRetrieval_train.json")["train"]
    fr_val_dataset = load_dataset('json', data_files="data/splits/FrDoc2BotRetrieval_val.json")["train"]


    vi_train_dataset = load_dataset('json', data_files="data/splits/ViDoc2BotRetrieval_train.json")["train"]
    vi_val_dataset = load_dataset('json', data_files="data/splits/ViDoc2BotRetrieval_val.json")["train"]

    train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]
    val_dataset = [y for dataset in [fr_val_dataset, vi_val_dataset] for y in dataset]



    train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

    all_passages = []
    for file_name in ['fr', 'vi']:
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)

    train(args, train_dataset, val_dataset, all_passages)
    
def train(args, train_dataset, val_dataset, all_passages):


    Eq = "sentence-transformers/LaBSE"
    Ep = "sentence-transformers/LaBSE"

    model_args = RetrievalArgs()
    model_args.warmup_rate = 0.1
    model_args.weight_decay = 0.1
    model_args.batch_size = 32
    model_args.learning_rate = 2e-5
    model_args.num_train_epochs = 50
    model_args.embed_batch_size = 32
    model_args.hard_negatives = True
    model_args.retrieve_n_docs = 20
    model_args.accumulation_steps = 4
    model_args.include_title = False



   

    cuda_available = torch.cuda.is_available()
    model = RetrievalModel(
        model_type = "dpr",
        context_encoder_name = Ep,
        query_encoder_name = Eq,
        args = model_args,
        use_cuda = cuda_available
    )
    #breakpoint()
    #rename columns of train_dataset to query_text, gold_passage, hard_negative 
    train_dataset = pd.DataFrame(train_dataset)
    train_dataset = train_dataset.rename(columns={'query': 'query_text', 'positive': 'gold_passage', 'negative': 'hard_negative'})

    model.train_model(train_dataset, eval_data = val_dataset, \
                output_dir = './model/', \
                show_running_loss = True)
    model.eval_model(test, verbose=True)

    '''model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    query_tokenized = tokenizer(train_dataset['query'], padding=True, truncation=True, return_tensors='pt')
    passage_tokenized = tokenizer(all_passages, padding=True, truncation=True, return_tensors='pt')

    query_embeddings = model(**query_tokenized).pooler_output
    passage_embeddings = model(**passage_tokenized).pooler_output


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(10):
        for batch in train_loader:
            optimizer.zero_grad()
            query_embeddings = model(**tokenizer(batch['query'], padding=True, truncation=True, return_tensors='pt')).pooler_output
            passage_embeddings = model(**tokenizer(all_passages, padding=True, truncation=True, return_tensors='pt')).pooler_output
            loss = torch.nn.CosineSimilarity()(query_embeddings, passage_embeddings)
            loss.backward()
            optimizer.step()
            print(loss)'''
    






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sentence-transformers')
    args = parser.parse_args()
    main(args)