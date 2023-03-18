from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json
import pandas as pd
from rank_bm25 import BM25Okapi
import jieba
from datasets import Dataset
import argparse
from tqdm import tqdm
from config import hparams
hp = hparams()




class BM25Search:
    """A class for searching documents using BM25.

    Args:
        documents (list): A list of documents to search.
        language (str, optional): The language of the documents. Defaults to "english".

    Attributes:
        documents (list): A list of documents to search.
        language (str): The language of the documents.
        tokenized_docs (list): A list of tokenized documents.
        bm25 (BM25Okapi): A BM25Okapi model for ranking documents.
    """

    def __init__(self, documents, language="english"):
        self.documents = documents
        self.language = language
        self.tokenized_docs = self.tokenize_documents()
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def tokenize_documents(self):
        """Tokenizes the documents using the appropriate method based on the language.

        Returns:
            list: A list of tokenized documents.
        """
        if self.language == "chinese":
            tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
        else:
            tokenized_docs = [doc.split(" ") for doc in self.documents]
        return tokenized_docs

    def search(self, query, n=10):
        """Searches for the most relevant documents to a query using BM25.

        Args:
            query (str): The query to search for.
            n (int, optional): The number of top documents to return. Defaults to 10.

        Returns:
            list: A list of the top `n` most relevant documents.
        """
        if self.language == "chinese":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        sorted_doc_scores = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
        top_docs = [self.documents[i[0]] for i in sorted_doc_scores[:n]]
        return top_docs



def main(args):
    for language in args.languages:
        lang_paths =  hp.lang_data_paths[language]

        if hp.lang_data_paths[language]["preprocessed_available"]:
            # french and vietnamese
            for stage in lang_paths:
                if stage == "short_name":
                    continue
                dataset = MsDataset.load(lang_paths[stage]["modelscope_url"], download_mode=DownloadMode.FORCE_REDOWNLOAD)
                hf_dataset  = dataset.to_hf_dataset().train_test_split(test_size = hp.test_size, seed = hp.seed)
                hf_train_dataset = hf_dataset['train']
                hf_val_dataset = hf_dataset['test']
                os.makedirs(os.path.dirname(lang_paths[stage]['path']), exist_ok=True)
                hf_train_dataset.to_json(f"{lang_paths[stage]['path']}_train.json")
                hf_val_dataset.to_json(f"{lang_paths[stage]['path']}_val.json")
        else:
            # english and chinese
            # 1. downloaded and saved provided json.
            dataset = MsDataset.load(lang_paths["modelscope_url"], download_mode=DownloadMode.FORCE_REDOWNLOAD)
            hf_dataset = dataset.to_hf_dataset()
            os.makedirs(os.path.dirname(lang_paths['default_data_path']), exist_ok=True)
            hf_dataset.to_json(f"{lang_paths['default_data_path']}.json")
            print("line 93")
            # 2. generate passages.csv for that language
            all_passages = []
            all_ids = []

            for i in range(len(hf_dataset)):
                passage = eval(hf_dataset['passages'][i])
                all_passages += passage
                all_ids += [i] * len(passage)
            print("line 102")
                
            all_passages_df = pd.DataFrame({"passage": all_passages, 'ids': all_ids})
            # all_passages_df["passage"].to_csv(os.path.join(hp.data_dir, f"{language}_passages.csv"), header=False, index=False)
            all_passages_df.to_csv(os.path.join(hp.data_dir, "all_passages", f"{language}_passages_w_ids.csv"))
            print("line 107")
            with open(os.path.join(hp.data_dir, "all_passages", f"{lang_paths['short_name']}.json"), "w") as f:
                f.write(json.dumps(all_passages, indent=4))
            print(f"Number of passages extracted for {language} dialogue data = {len(all_passages)}.")
            breakpoint()
            bm25 = BM25Search(documents=all_passages_df["passage"], language=language)

             # 3. generate retriever data
            retriever_data = {"query": [], "positive":[], "negative":[]}
            for i in tqdm(range(len(hf_dataset))):
                passage = eval(hf_dataset['passages'][i])
                query = hf_dataset['query'][i]
                docs = bm25.search(query, n=hp.num_bm25_docs)
                positive = negative = ""
                for doc in docs:
                    if doc in passage:
                        positive = doc
                    else:
                        negative = doc
                    if positive != "" and negative != "":
                        break
                if positive == "":
                    positive = passage[0]
                retriever_data["query"].append(query)
                retriever_data["positive"].append(positive)
                retriever_data["negative"].append(negative)
            retriever_hf_ds = Dataset.from_dict(retriever_data)
            retriver_splits = retriever_hf_ds.train_test_split(test_size=hp.test_size, seed=hp.seed)
            retriver_splits["train"].to_json(f"{lang_paths['retrieval']['path']}_train.json")
            retriver_splits["test"].to_json(f"{lang_paths['retrieval']['path']}_val.json")

            # 4. generate rerank data

            # 5. generate generation data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--languages', nargs='+', default=hp.available_languages)
    args = parser.parse_args()
    main(args)