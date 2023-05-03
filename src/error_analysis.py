import pandas as pd
import numpy as np
import os
import shutil
import json
import argparse
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, BertModel
from config import hparams
import torch
from sklearn.decomposition import PCA
from utils.huggingface_retriever import DocumentGroundedDialogRetrievalTrainerHF, HFDPRModel, DocumentGroundedDialogRetrievalPreprocessorHF
from utils.labse_retriever import DocumentGroundedDialogRetrievalTrainerLabse
from utils.labse_retriever import LabseDPRModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


hp = hparams()


def load_data(args):
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
    if "french_2k_raw" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['french_2k_raw']['stages']['retrieval']['path']}_val.json")['train'])
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
    if "vietnamese_2k_raw" in args.languages:
        temp_datasets.append(load_dataset(
            'json', data_files=f"{lang_data_paths['vietnamese_2k_raw']['stages']['retrieval']['path']}_val.json")['train'])
        passage_languages.append('vi')

    eval_dataset = [x for dataset in temp_datasets for x in dataset]

    all_passages = []
    for file_name in set(passage_languages):
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)

    return eval_dataset, all_passages


def infer_files(args):
    eval_dataset, all_passages = load_data(args)

    if args.model_type == "labse":
        model_save_path = args.model_checkpoint
        if args.model_checkpoint is None:
            raise Exception("Model checkpoint cannot be empty for LaBSE")
        trainer = DocumentGroundedDialogRetrievalTrainerLabse(
            model_save_path=model_save_path,
            train_dataset=None,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
        trainer.evaluate(checkpoint_path=args.model_checkpoint)

        # evaluate_results.json should be created
        src_eval_result_fp = os.path.join(os.path.dirname(
            args.model_checkpoint), "evaluate_result.json")
        dst_folder = os.path.dirname(args.output_file_path)
        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy(src_eval_result_fp, dst_folder)
        os.rename(os.path.join(dst_folder, "evaluate_result.json"),
                  args.output_file_path)


def generate_embeddings(args):
    if args.model_type == 'mbert':
        model_name = "bert-base-multilingual-cased"
        model = HFDPRModel(model_name)
    else:
        model_name = "setu4993/LaBSE"
        model = LabseDPRModel()

    model.load_state_dict(torch.load(args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset, all_passages = load_data(args)
    #eval_dataset = eval_dataset[0:10]
    languages = []
    domains = []
    embeddings = np.empty((0, 768), dtype=float)
    model.eval()
    for i in tqdm(eval_dataset):
        query = i['query']
        passage = i['positive']
        language = ((passage.split("//")[-1]).split("-")[0]).strip()
        domain = ((passage.split("//")[-1]).split("-")[1]).strip()
        with torch.no_grad():
            #tokenized_input = tokenizer(query, return_tensors='pt')
            tokenized_input = tokenizer(
                query,
                padding=True,
                return_tensors='pt',
                max_length=512,
                # max_length=self.query_sequence_length,
                truncation=True)
            #outputs = model(**tokenized_input, output_hidden_states=True)
            #last_hidden_states = outputs.hidden_states[-1]
            #cls_embedding = last_hidden_states [0,0,:].cpu().numpy().reshape(1,-1)
            embedding = model.encode(model.qry_encoder, tokenized_input.input_ids,
                                     tokenized_input.attention_mask)
        languages.append(language)
        domains.append(domain)
        embeddings = np.append(embeddings, embedding, axis=0)
        # print(cls_embedding.shape)
        #embeddings = np.append(embeddings, cls_embedding, axis=0)
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    new_embeddings = pca.transform(embeddings)
    df_dict = {'language': languages, 'domain': domains,
               'dim1': new_embeddings[:, 0].tolist(), 'dim2': new_embeddings[:, 1].tolist()}
    df = pd.DataFrame(df_dict)
    df.to_csv('{}.csv'.format(args.model_type), index=False)
    # print(df)
    # return df


def plot(args):
    df = pd.read_csv(args.embeddings_file)
    x = df['dim1']
    y = df['dim2']
    lang_to_col = {'fr': 'blue', 'vi': 'red'}
    domain_to_col = {'Health': 'red', 'Technology': 'blue', 'VeteransAffairs': 'green', 'HealthCareServices': 'yellow', 'Insurance': 'pink',
                     'PublicServices': 'black', 'SocialSecurity': 'brown', 'DepartmentOfMotorVehicles': 'purple', 'StudentFinancialAidinUSA': 'orange'}

    plt.scatter(x, y, c=df['language'].map(lang_to_col))
    plt.title('Language Specific Embeddings')
    handles = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color,
                        label=label)[0] for label, color in lang_to_col.items()]

    plt.legend(handles=handles, loc='upper right')
    plt.savefig('lang.png')
    plt.clf()

    plt.scatter(x, y, c=df['domain'].map(domain_to_col))
    plt.title('Domain Specific Embeddings')
    handles = [plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color,
                        label=label)[0] for label, color in domain_to_col.items()]

    #plt.legend(handles=handles, loc='upper right')
    plt.savefig('domain.png')


def len_analysis(args):
    bert_results = json.load(open(args.bert_file))
    labse_results = json.load(open(args.labse_file))
    eval_dataset, all_passages = load_data(args)
    predictions = []
    lengths = []
    total_lens = {}
    error_lens = {}
    for i in range(0, len(bert_results['targets'])):
        ground_truth = bert_results['targets'][i]
        bert_pred = bert_results['outputs'][i][0]
        query_len = len(eval_dataset[i]['query'])
        len_key = query_len//50

        if len_key not in total_lens.keys():
            total_lens[len_key] = 1
        else:
            total_lens[len_key] += 1

        if ground_truth != bert_pred:
            if len_key not in error_lens.keys():
                error_lens[len_key] = 1
            else:
                error_lens[len_key] += 1

    error_percentages = {}
    for key in error_lens.keys():
        error_percentages[key] = error_lens[key]/total_lens[key]
    error_percentages = dict(sorted(error_percentages.items()))
    total_lens = dict(sorted(total_lens.items()))
    # plt.plot(lengths,predictions)
    # plt.savefig('lengths.png')
    print(total_lens)
    print(error_percentages)


def main(args):
    model_results = json.load(open(args.bert_file))
    lang_errors = {}
    domain_errors = {}
    total_lang = {}
    total_domain = {}
    for i in range(0, len(model_results['targets'])):
        ground_truth = model_results['targets'][i]
        bert_pred = model_results['outputs'][i][0]
        '''labse_pred = labse_results['outputs'][i][0]
        if ground_truth == labse_pred  and ground_truth!=bert_pred:
            errors.append(i)'''
        language = ((ground_truth.split("//")[-1]).split("-")[0]).strip()
        domain = ((ground_truth.split("//")[-1]).split("-")[1]).strip()
        if ground_truth != bert_pred:
            if language not in lang_errors:
                lang_errors[language] = 1
            else:
                lang_errors[language] += 1

            if domain not in domain_errors:
                domain_errors[domain] = 1
            else:
                domain_errors[domain] += 1

        if language not in total_lang:
            total_lang[language] = 1
        else:
            total_lang[language] += 1

        if domain not in total_domain:
            total_domain[domain] = 1
        else:
            total_domain[domain] += 1

    '''print('errors length ',len(errors))
    
    print('first 5 errors')
    count =0
    for i in errors:
        print('Ground Truth: ', model_results['targets'][i])
        print('BERT Predicted: ', model_results['outputs'][i][0])
        print('LABSE Predicted: ', labse_results['outputs'][i][0])
        print()
        print()
        count = count+1
        if count>5:
            break'''
    language_error_percentage = {}
    domain_error_percentage = {}
    for i in total_lang.keys():
        language_error_percentage[i] = (lang_errors[i]/total_lang[i])*100

    for i in total_domain.keys():
        domain_error_percentage[i] = (domain_errors[i]/total_domain[i])*100

    '''print('Total lang ', total_lang)
    print('lang error ', lang_errors)
    print('Total domain ', total_domain)
    print('domain error ', domain_errors)'''

    print('Language Error Percentage ', language_error_percentage)
    print('Domain Error Percentage ', domain_error_percentage)


def compute_stats(args):
    stats_dict = {}
    predictions_accuracy_dict_flat = {}
    predictions_accuracy_dict_nested = {}
    for language in ["french", "vietnamese"]:
        predictions_accuracy_dict_flat[language] = {}

        for model_train_dataset_type in ["2k_raw", "gpt", "conqrr_rewritten_true", "conqrr_rewritten_false"]:
            for eval_dataset_type in ["2k_raw", "gpt"]:
                dict_key = f"[\"{model_train_dataset_type}\", \"{eval_dataset_type}\", \"{language}\"]"
                results_fp = f"/usr0/home/sohamdit/Multilingual_QA/src/results/english_{model_train_dataset_type}_labse/{language}_{eval_dataset_type}.json"
                args_copy = vars(args).copy()
                args_copy['languages'] = [f"{language}_{eval_dataset_type}"]
                eval_dataset, all_passages = load_data(args)
                model_results = json.load(open(results_fp))
                lang_errors = {}
                domain_errors = {}
                total_lang = {}
                total_domain = {}
                total_lens = {}
                error_lens = {}
                correct_predictions = []
                for i in range(len(model_results['targets'])):
                    ground_truth = model_results['targets'][i]
                    model_pred = model_results['outputs'][i][0]
                    short_language = ((ground_truth.split("//")
                                       [-1]).split("-")[0]).strip()
                    domain = ((ground_truth.split("//")
                              [-1]).split("-")[1]).strip()
                    correct_predictions.append(
                        1 if ground_truth == model_pred else 0)
                    if ground_truth != model_pred:
                        if short_language not in lang_errors:
                            lang_errors[short_language] = 1
                        else:
                            lang_errors[short_language] += 1

                        if domain not in domain_errors:
                            domain_errors[domain] = 1
                        else:
                            domain_errors[domain] += 1

                    if short_language not in total_lang:
                        total_lang[short_language] = 1
                    else:
                        total_lang[short_language] += 1

                    if domain not in total_domain:
                        total_domain[domain] = 1
                    else:
                        total_domain[domain] += 1

                    query_len = len(eval_dataset[i]['query'])
                    len_key = query_len//50

                    if len_key not in total_lens.keys():
                        total_lens[len_key] = 1
                    else:
                        total_lens[len_key] += 1

                    if ground_truth != model_pred:
                        if len_key not in error_lens.keys():
                            error_lens[len_key] = 1
                        else:
                            error_lens[len_key] += 1

                language_error_percentage = {}
                domain_error_percentage = {}
                len_error_percentage = {}
                for i in total_lang.keys():
                    language_error_percentage[i] = (
                        lang_errors[i]/total_lang[i])*100

                for i in total_domain.keys():
                    if i in domain_errors:
                        domain_error_percentage[i] = (
                            domain_errors[i]/total_domain[i])*100
                    else:
                        domain_error_percentage[i] = 0

                for key in error_lens.keys():
                    len_error_percentage[key] = error_lens[key]/total_lens[key]

                if model_train_dataset_type not in stats_dict:
                    stats_dict[model_train_dataset_type] = {}
                    predictions_accuracy_dict_nested[model_train_dataset_type] = {}

                if eval_dataset_type not in stats_dict[model_train_dataset_type]:
                    stats_dict[model_train_dataset_type][eval_dataset_type] = {}
                    predictions_accuracy_dict_nested[model_train_dataset_type][eval_dataset_type] = {}

                stats_dict[model_train_dataset_type][eval_dataset_type][language] = {
                    f"language_error_percentage": language_error_percentage,
                    f"domain_error_percentage": domain_error_percentage,
                    f"len_error_percentage": dict(sorted(len_error_percentage.items())),
                }

                predictions_accuracy_dict_flat[language][dict_key] = correct_predictions
                predictions_accuracy_dict_nested[model_train_dataset_type][eval_dataset_type][language] = correct_predictions

    french_df = pd.DataFrame.from_dict(
        predictions_accuracy_dict_flat['french'])
    vietnamese_df = pd.DataFrame.from_dict(
        predictions_accuracy_dict_flat['vietnamese'])

    french_df.to_csv(os.path.join(args.stats_output_dir,
                     'french_correct_predictions.csv'))
    vietnamese_df.to_csv(os.path.join(
        args.stats_output_dir, 'vietnamese_correct_predictions.csv'))

    with open(os.path.join(args.stats_output_dir, 'stats_dict.json'), 'w') as f:
        json.dump(stats_dict, f)

    with open(os.path.join(args.stats_output_dir, 'predictions_accuracy_dict_nested.json'), 'w') as f:
        json.dump(predictions_accuracy_dict_nested, f)

    with open(os.path.join(args.stats_output_dir, 'predictions_accuracy_dict_flat.json'), 'w') as f:
        json.dump(predictions_accuracy_dict_flat, f)

    print(f'Saved all dictionaries and csv files')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=[
                        "infer_files", "compute_stats", "main", "generate_embeddings", "plot", "len_analysis"])
    parser.add_argument("-mt", "--model_type", type=str, choices=["labse"])
    parser.add_argument("-mc", "--model_checkpoint", type=str, required=False)
    parser.add_argument("-ofp", "--output_file_path", type=str, required=False)
    parser.add_argument("-sod", "--stats_output_dir", type=str, required=False)
    parser.add_argument('-bf', '--bert_file', type=str,
                        default="/home/sgowrira/Multilingual_QA/src/soham_models/retriever/mbert-hf/finetuned_vi_fr/evaluate_result.json")
    parser.add_argument('-lf', '--labse_file', type=str,
                        default="/home/sgowrira/Multilingual_QA/src/soham_models/retriever/labse/finetuned_vi_fr/evaluate_result.json")
    parser.add_argument('-cp', '--checkpoint', type=str,
                        default='/home/sgowrira/Multilingual_QA/src/soham_models/retriever/mbert-hf/finetuned_vi_fr/model_weights.bin')
    parser.add_argument("-l", '--languages', nargs='+',
                        default=["french", "vietnamese"])
    # parser.add_argument('-m', '--model', type=str, default = 'mbert')
    parser.add_argument('-ef', '--embeddings_file', type=str,
                        default="/home/sgowrira/Multilingual_QA/src/mbert.csv")
    args = parser.parse_args()

    if args.mode == "main":
        main(args)
    elif args.mode == "infer_files":
        infer_files(args)
    elif args.mode == "generate_embeddings":
        generate_embeddings(args)
    elif args.mode == "generate_embeddings":
        plot(args)
    elif args.mode == "generate_embeddings":
        len_analysis(args)
    elif args.mode == "compute_stats":
        compute_stats(args)
    else:
        print(f"Such a function is not supported")


# GPT wrong and Raw 2k model correct
# manual inspection above of
# group by categories if interesting

# GPT right and raw 2k model wrong
# same as above
