import argparse
from datasets import load_dataset
import random
import torch
import numpy as np
import json

def main(args):
    for name in ['EnDoc2BotRetrieval_train', 'ViDoc2BotRetrieval_val', 'FrDoc2BotRetrieval_val']:
        dataset = load_dataset('json', data_files=f"./data/splits/{name}.json")['train']
        
        if name == 'EnDoc2BotRetrieval_train':
            random_indices = random.sample(range(len(dataset)), 2000)
        else:
            random_indices = random.sample(range(len(dataset)), 500)
        random_dataset = dataset.select(random_indices)
        
        random_dataset.to_json(f'./data/gpt_splits/{name}.json')
        
def read_splits(args):
    en_train_file = load_dataset('json', data_files='./data/gpt_splits/EnDoc2BotRetrieval_train.json')["train"]
    print(len(en_train_file))
        
                            
    
    
    
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('-ak','--api_key', type=str)
    args=parser.parse_args()
    
    #set all seeds for reproducability
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #main(args)
    read_splits(args)