import os

def main():
    for language in ["french", "vietnamese"]:
        # for model_train_dataset_type in ["2k_raw", "gpt"]:
        for model_train_dataset_type in ["conqrr_rewritten_true", "conqrr_rewritten_false"]:
            for eval_dataset_type in ["2k_raw", "gpt"]:
                cmd = f"CUDA_VISIBLE_DEVICES=0 python3 error_analysis.py -m infer_files -mt labse -mc /usr0/home/sohamdit/Multilingual_QA/src/finetuned_models/retriever/labse/pretrained_english_{model_train_dataset_type}/model_weights.bin -ofp /usr0/home/sohamdit/Multilingual_QA/src/results/english_{model_train_dataset_type}_labse/{language}_{eval_dataset_type}.json -l {language}_{eval_dataset_type}"

                print("Starting evaluating...")
                print(cmd)
                os.system(cmd)

                print(f"Done with: model_train_dataset_type - {model_train_dataset_type}; eval_dataset_type - {eval_dataset_type}  eval lang - {language}")
                print("*" * 20)

if __name__ == '__main__':
    main()