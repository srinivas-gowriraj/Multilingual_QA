from transformers import AutoTokenizer, AutoModel
import torch
import jieba
from datasets import load_dataset
from tqdm import tqdm
import argparse

class SentenceSimilarity:
    def __init__(self, language='english', model_name='bert-base-multilingual-cased'):
        """
        Initializes the SentenceSimilarity object.
        
        Args:
            model_name (str): The name of the pre-trained model to use for generating sentence embeddings.
                              Defaults to 'bert-base-multilingual-cased'.
        """
        self.language = language
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        
    def tokenize(self, text):
        """
        Tokenizes text using the appropriate tokenizer for the pre-trained model's language.
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            The tokenized text as a list of strings.
        """
        if self.language == 'chinese':
            return list(jieba.cut(text))
        else:
            return self.tokenizer.tokenize(text)

    def cosine_similarity(self, s1, s2):
        """
        Computes the cosine similarity between two sentences using sentence embeddings.
        
        Args:
            s1 (str): The first sentence to compare.
            s2 (str): The second sentence to compare.
            
        Returns:
            The cosine similarity between the two sentences as a float between 0 and 1.
        """
        # Tokenize the input sentences and convert to tensors
        # s1_tokens = self.tokenize(s1)
        # s2_tokens = self.tokenize(s2)
        inputs_0 = self.tokenizer(s1, return_tensors='pt', padding=True, truncation=True)
        inputs_1 = self.tokenizer(s2, return_tensors='pt', padding=True, truncation=True)
        # inputs = self.tokenizer(s1_tokens, s2_tokens, return_tensors='pt', padding=True, truncation=True)
        
        # Generate sentence embeddings for the two sentences
        with torch.no_grad():
            embeddings = [self.model(**inputs_0).pooler_output, self.model(**inputs_1).pooler_output]
        
        # Compute the cosine similarity between the embeddings
        similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1]).item()
        
        return similarity

def test_similarity_for_retrieval_data(language, path):
    ds = load_dataset('json', data_files=path)['train']
    sent_sim = SentenceSimilarity(language=language)
    all_scores = []
    num_pos_greater = 0
    num_neg_greater = 0
    for i in tqdm(range(len(ds))):
        query = ds[i]['query']
        positive = ds[i]['positive']
        negative = ds[i]['negative']
        pos_score = sent_sim.cosine_similarity(query, positive)
        neg_score = sent_sim.cosine_similarity(query, negative)
        all_scores.append((pos_score, neg_score))
        if pos_score > neg_score:
            num_pos_greater += 1
        else:
            num_neg_greater += 1

    print(f"Number of total sample for {language}. {len(all_scores)}")

    print(f"Number of pos>neg: {num_pos_greater}")
    print(f"Percentage pos>neg: {num_pos_greater/len(all_scores)*100}%")

    print(f"Number of pos<neg: {num_neg_greater}")
    print(f"Percentage pos<neg: {num_neg_greater/len(all_scores)*100}%")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', help='Language of input data', default="english", choices=["english", "french", "vietnamese", "chinese"], type=str)
    parser.add_argument('-p', '--path', help='File path to input json file for retriever', required=True, type=str)
    args = parser.parse_args()

    test_similarity_for_retrieval_data(args.lang, args.path)
