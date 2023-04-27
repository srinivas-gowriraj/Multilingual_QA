import openai
import asyncio
from typing import Any
import argparse
import os
import json
from tqdm import tqdm
import time

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def load_data(args):
    if args.language == "en":
        file_path = "./data/gpt_splits/EnDoc2BotRetrieval_train.json"
    elif args.language == "fr":
        file_path = "./data/gpt_splits/FrDoc2BotRetrieval_val.json"
    elif args.language=="vi":
        file_path = "./data/gpt_splits/ViDoc2BotRetrieval_val.json"
    else:
        raise Exception("Invalid language selection")
        
    queries = [json.loads(item)['query'] for item in open(file_path,'r').readlines()]
    positives = [json.loads(item)['positive'] for item in open(file_path,'r').readlines()]
    negatives = [json.loads(item)['negative'] for item in open(file_path,'r').readlines()]
    prompts=[]
    
    print('Creating prompts')
    for i in tqdm(range(len(queries))):
        query = queries[i]
        question = query.replace('<last_turn>', 'Question:').replace('<user>','').split('<agent>')[0].lstrip('Question:').strip()
        
        question_index = query.find(question)
        
        context = query[question_index+len(question):].strip()
        
        
        if args.language == "en":
            prompt = f"""Rewrite the question into a informative query explicitly mentioning important relevant details from the provided context.

Context: {context}

Question: {question}

Re-Written Question:
"""
        elif args.language == "vi":
            prompt = f"""Viết lại câu hỏi thành một truy vấn thông tin đề cập rõ ràng các chi tiết quan trọng có liên quan từ ngữ cảnh được cung cấp.

Bối cảnh: {context}

Câu hỏi: {question}

Viết lại câu hỏi:
"""
        elif args.language == "fr":
            prompt = f"""Réécrivez la question dans une requête informative mentionnant explicitement les détails pertinents importants du contexte fourni.
            
Contexte: {context}

Question: {question}

Question réécrite :
"""
        prompts.append(prompt)
        
    return prompts, positives,negatives
        
            

            
        
    
    
def main(args):
    openai.api_key = args.api_key
    prompts, positives,negatives = load_data(args)
    
    # messages_list=[]
    # for i in prompts:
    #     messages_list.append([{"role": "user", "content":i}])
    
    # predictions = asyncio.run(
    #     dispatch_openai_requests(
    #         messages_list=messages_list,
    #         model="gpt-3.5-turbo",
    #         temperature=0.0,
    #     )
    # )
    print('started sending requests to openAI')
    predictions =[]
    for i in tqdm(prompts):
        max_retries = 5
        retry_count = 0
        response = None
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": i,}
                    ],
                    temperature=0.2,
                )
                break
            except Exception as e:
                print(f"Request failed with error: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count) 
        if response is None:
            print(f"Request failed after maximum retries for {i}")
        else:
            #output = response['choices'][0]['message']['content']
            predictions.append(response)
        
        #open-ai can handle only 3 queries per min
        time.sleep(22)
        
        
        
    
    updated_queries = []
    for i, x in enumerate(predictions):
        #print(f"Response {i}: {x['choices'][0]['message']['content']}\n\n")
        query = x['choices'][0]['message']['content']
        updated_queries.append(query)
        
    if args.language == "en":
        filename = "/home/sgowrira/Multilingual_QA/src/data/gpt_splits/EnDoc2BotRetrieval_train_gpt.json"
    elif args.language == "fr":
        filename = "/home/sgowrira/Multilingual_QA/src/data/gpt_splits/FrDoc2BotRetrieval_val_gpt.json"
    elif args.language == "vi":
        filename = "/home/sgowrira/Multilingual_QA/src/data/gpt_splits/ViDoc2BotRetrieval_val_gpt.json"
        
        
    with open(filename, 'w') as f:
        for i in range(len(updated_queries)):
            dictionary = dict()
            dictionary["query"] = updated_queries[i]
            dictionary["positive"] = positives[i]
            dictionary["negative"] = negatives[i]

            f.write(json.dumps(dictionary))
            f.write('\n')
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ak', '--api_key', type=str, required=True)
    parser.add_argument('-l', '--language', type=str)
    args=parser.parse_args()
    
    main(args)
    