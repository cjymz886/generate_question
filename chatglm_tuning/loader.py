import json
import random

def convert_data():
    data = []
    with open(r'E:\mystudy\T5-Finetuning-PyTorch\corpus\prompt_data\prompt_data.txt' ,'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            line['answer'] = line['answer'] +'</s>'
            data.append(line)
    random.shuffle(data)

    with open('./data/prompt_data.txt', 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d,ensure_ascii=False)+'\n')


convert_data()
