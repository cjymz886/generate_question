
from transformers import AutoTokenizer,AutoModel
import torch
from peft import get_peft_model, LoraConfig, TaskType



model = AutoModel.from_pretrained(
    "yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True).half().cuda()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value',],
)

model = get_peft_model(model, peft_config)

# 在这里加载lora模型，注意修改chekpoint
peft_path = r"E:\openlab\MyLLM\save_models\checkpoint-4000\chatglm-lora.pt"
model.load_state_dict(torch.load(peft_path), strict=False)
tokenizer = AutoTokenizer.from_pretrained("yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True)


# text ="帮我生成10条包含[杭州,西湖]关键词的问句"
# text = "请告诉我包含[微信,支付]关键词的问句有哪些"
# text = "跟[比亚迪的车怎样]相似问句有哪些"
# text = "帮我生成12条与[一体机适不适合打游戏]相似的问句"
text = "帮我生成12条与[请问员工工作进度落后,管理者应该怎么办]相关的问题"
# text = "[美国CPI为何暴涨]问句包含哪些关键词"
# with torch.autocast("cuda"):
#     res, history = model.chat(tokenizer=tokenizer, query=text,max_length=256)
#     print(res)
#
# exit()


with torch.autocast("cuda"):
    ids = tokenizer.encode(text)
    input_ids = torch.LongTensor([ids]).to('cuda')
    out = model.generate(
        input_ids=input_ids,
        max_length=200,
        do_sample=True,
        top_p=0.65,
        temperature=0.95,
        # top_k=40,
        eos_token_id=150005,
        repetition_penalty = 1.3,
        # length_penalty = 1.0,
    )


    out_text = tokenizer.decode(out[0])
    print(text)
    # print(out_text)
    # answer = out_text.replace(text, "").replace("\nEND", "").strip().split('\n\n')[0]
    answer = out_text.replace(text, "").strip().split('\n\n')[0]
    print(answer)
    # end_index = -1
    # last_sep = -1
    # for i, item in enumerate(answer):
    #     if item == '[MASK]':
    #         text[i] = ''
    #     elif item == '[CLS]':
    #         text[i] = '\n'
    #     elif item == '\n\n':
    #         last_sep = i
    #         break
    #     elif item == '</s>':
    #         end_index = i
    #         break
    # print(end_index, last_sep)
    # answer = answer[:end_index]
    # answer = answer[:last_sep]
    # print(answer)