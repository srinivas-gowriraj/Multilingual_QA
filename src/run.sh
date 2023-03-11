(python train_retrieval.py && echo "Train retrieval done") > 1_output_train_retrieval.txt

(python train_rerank.py && echo "Train rerank done") > 2_output_train_rerank.txt

(python train_generation.py && echo "Train generation done") > 3_output_train_generation.txt

(python inference_retrieval.py && echo "inference retrieval done") > 4_output_inference_retrieval.txt

(python inference_rerank.py && echo "inference rerank done") > 5_output_inference_rerank.txt

(python inference_generation.py && echo "inference generation done") > 6_output_inference_generation.txt
