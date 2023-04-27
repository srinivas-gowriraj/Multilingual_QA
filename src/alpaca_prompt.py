import json
import os
import llm_client
from tqdm import tqdm

print('file processing started')
data_dir = '/home/sgowrira/Multilingual_QA/data/splits'
language = 'Zh'
split = 'train'
shot = 'few'
# language = 'Vi'
# split = 'val'
file_path = os.path.join(data_dir, f'{language}Doc2BotRetrieval_{split}.json')
queries = [json.loads(item)['query'] for item in open(file_path,'r').readlines()]
positive = [json.loads(item)['positive'] for item in open(file_path,'r').readlines()]
negative = [json.loads(item)['negative'] for item in open(file_path,'r').readlines()]
alpaca_client = llm_client.Client(address="tir-1-7")
updated_queries = []


#for i in range(1):
for i in tqdm(range(len(queries))):
    prompt = queries[i]
    
    original_prompt = prompt
    
    if language == 'En':
        prompt_old = prompt.replace('<last_turn>', 'Question:').replace('<user>','').split('<agent>')
        original_question = prompt_old[0].strip('Question:')
        # prompt = 'context:'.join([prompt[0], ''.join(prompt[1:])]).replace("  ", " ")
        prompt = f"input: {prompt} output:"
        if shot == 'zero':
            #prompt = 'context:'.join([prompt_old[0], ''.join(prompt_old[1:])]).replace("  ", " ")
            task_prompt = " Based on the dialogue history, paraphrase the <last_turn> utterance such that it asks question with all important details are explicitly mentioned. "
        else:
            task_prompt = """Based on the dialogue history, paraphrase the <last_turn> utterance such that it asks question with all important details are explicitly mentioned.
input: “<last_turn> what was his legacy <agent> he is an actor <user> who is cary grant ? “
output: “what was cary grant’s legacy ?”
input: “<last_turn> Who won ?<agent> MasterChef season 10 aired May 29 to September 18, 2019. <user> When did the season air? <agent> Sarah Faherty was the last person eliminated …  <user> Who was the last person eliminated from Masterchef season 10?”
output: Who won MasterChef season 10?"""
    
    elif language == 'Zh':
        prompt_old = prompt.replace('<last_turn>', '问题:').replace('<user>','').split('<agent>')
        original_question = prompt_old[0].strip('问题:')
        # prompt = "上下文:".join([prompt[0], ''.join(prompt[1:])]).replace("  ", " ")
        prompt = f"输入: {prompt} 输出:"
        if shot == 'zero':
            #prompt = '上下文:'.join([prompt_old[0], ''.join(prompt_old[1:])]).replace("  ", " ")
            #task_prompt = "根据上下文，只改写问题，使得所有重要细节都被明确提及。"
            task_prompt = "根据对话历史，解释 <last_turn> 话语，使其提出问题并明确提及所有重要细节。"
        else:
            task_prompt = """根据对话历史，解释 <last_turn> 话语，使其提出问题并明确提及所有重要细节。
输入：“<last_turn> 他的遗产是什么 <agent> 他是演员 <user> 谁是 卡里格兰特？ “
输出：“加里格兰特的遗产是什么？”
输入：“<last_turn> 谁赢了？<agent> 大厨 第 10 季于 2019 年 5 月 29 日至 9 月 18 日播出。<user> 本季何时播出？ <agent> 莎拉·法赫蒂 是最后一个被淘汰的人...... <user> 谁是 大厨 第 10 季最后一个被淘汰的人？
输出：谁赢得了 大厨 第 10 季？"""

    elif language == 'Vi':
        prompt_old = prompt.replace('<last_turn>', 'Câu hỏi:').replace('<user>','').split('<agent>')
        original_question = prompt_old[0].strip('Câu hỏi:')
        # prompt = "Bối cảnh:".join([prompt[0], ''.join(prompt[1:])]).replace("  ", " ")
        prompt = f"đầu vào: {prompt} đầu ra"
        if shot == 'zero':
            #prompt = 'Bối cảnh:'.join([prompt_old[0], ''.join(prompt_old[1:])]).replace("  ", " ")
            #task_prompt = "Dựa trên bối cảnh, viết lại câu hỏi sao cho tất cả các chi tiết quan trọng được đề cập rõ ràng."
            task_prompt = "Dựa trên lịch sử đối thoại, diễn giải cách nói <last_turn> sao cho nó đặt câu hỏi với tất cả các chi tiết quan trọng được đề cập rõ ràng."
        else:
            task_prompt = """Dựa trên lịch sử đối thoại, diễn giải cách nói <last_turn> sao cho nó đặt câu hỏi với tất cả các chi tiết quan trọng được đề cập rõ ràng.
đầu vào: “<last_turn> di sản của anh ấy là gì <agent> anh ấy là một diễn viên <user> người được cấp cary ? “
đầu ra: "di sản của cary Grant là gì?"
đầu vào: “<last_turn> Ai đã thắng ?<agent> MasterChef mùa 10 phát sóng từ ngày 29 tháng 5 đến ngày 18 tháng 9 năm 2019. <user> Mùa phát sóng khi nào? <agent> Sarah Faherty là người cuối cùng bị loại … <user> Ai là người cuối cùng bị loại khỏi Masterchef mùa 10?”
đầu ra: Ai vô địch MasterChef mùa 10?"""
    elif language == 'Fr':
        prompt_old = prompt.replace('<last_turn>', 'Question:').replace('<user>','').split('<agent>')
        original_question = prompt_old[0].strip('Question:')
        # prompt = "Contexte:".join([prompt[0], ''.join(prompt[1:])]).replace("  ", " ")
        prompt = f"input: {prompt} sortie"
        if shot == 'zero':
            #prompt = 'Contexte:'.join([prompt_old[0], ''.join(prompt_old[1:])]).replace("  ", " ")
            #task_prompt = "Sur la base du contexte, réécrivez une question autonome de sorte que tous les détails importants soient explicitement mentionnés."
            task_prompt = "Sur la base de l'historique du dialogue, paraphrasez l'énoncé <last_turn> de manière à ce qu'il pose une question en mentionnant explicitement tous les détails importants."
        else:
            task_prompt = """ Sur la base de l'historique du dialogue, paraphrasez l'énoncé <dernier_tour> de manière à ce qu'il pose une question en mentionnant explicitement tous les détails importants.
input : "<last_turn> quel a été son héritage <agent> c'est un acteur <user> qui est cary grant ? "
sortie : "quel a été l'héritage de cary grant ?"
input : "<last_turn> Qui a gagné ?<agent> La saison 10 de MasterChef a été diffusée du 29 mai au 18 septembre 2019. <user> Quand la saison a-t-elle été diffusée ? <agent> Sarah Faherty a été la dernière personne éliminée…  <user> Qui a été la dernière personne éliminée de la saison 10 de Masterchef ?"
sortie : Qui a remporté la saison 10 de MasterChef ?"""
    

    
    
    #prompt = prompt + task_prompt  
    prompt = task_prompt + prompt   

    
    #breakpoint()
    if len(prompt) > 600:
        prompt = prompt[:600]
    #print(len(prompt))
    ouputs = alpaca_client.prompt(prompt, do_sample=False)
    new_question = ouputs[0].text.split(prompt)[1]
    updated_prompt = original_prompt.replace(original_question, new_question)
    updated_queries.append(updated_prompt)
    
    
    

with open(os.path.join(data_dir, f'{language}Doc2BotRetrieval_{split}_few_shot_paraphrased.json'), 'w') as f:
#with open(os.path.join(data_dir, f'{language}Doc2BotRetrieval_{split}_paraphrased.json'), 'w') as f:
   
    #for i in range(1):
    for i in range(len(queries)):
        dictionary = dict()
        dictionary["query"] = updated_queries[i]
        dictionary["positive"] = positive[i]
        dictionary["negative"] = negative[i]
        
        f.write(json.dumps(dictionary))
        f.write('\n')
    #f.write('\n'.join(updated_queries))




# prompt = queries[0]

# prompt = prompt.replace('<last_turn>', 'Question :').replace('<user>','').split('<agent>')
# prompt = 'context :'.join([prompt[0], ''.join(prompt[1:])]).replace("  ", " ")
# #task_prompt = " Based on the context, rewrite a stand alone question. "
# task_prompt = " Based on the context, paraphrase only the question such that all important details are explicitly mentioned. "
# prompt = prompt + task_prompt
                    
# ouputs = alpaca_client.prompt(prompt)
# breakpoint()
# print(ouputs[0].text.split(prompt)[1])