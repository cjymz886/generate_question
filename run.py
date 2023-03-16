import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Config():
        TRAIN_BATCH_SIZE = 200  # training batch size
        VALID_BATCH_SIZE = 200  # validation batch size
        TRAIN_EPOCHS = 5  # number of training epochs
        VAL_EPOCHS = 1 # number of validation epochs
        LEARNING_RATE = 5e-5  # learning rate
        MAX_SOURCE_TEXT_LENGTH = 10  # max length of source text
        MAX_TARGET_TEXT_LENGTH = 32  # max length of target text
        SEED = 42  # set seed for reproducibility
        device = torch.device("cuda")

        t5_path = r"E:\pretraing_models\torch\mengzi_t5_base"
        config_path = t5_path + r'\config.json'
        checkpoint_path = t5_path + r'\pytorch_model.bin'


def load_data(inputfile):
    data = []
    with open(inputfile, encoding='utf-8') as f:
        for line in f:
            tmp = {}
            line = json.loads(line)
            keywords, query = line['keywords'], line['query']
            data.append({'keywords':keywords, 'query':query})
    # data = data[:2000]
    df = pd.DataFrame(data)
    return df



class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }



def train(epoch, tokenizer, model, device, loader, optimizer):

    model.train()
    with tqdm(total=loader.__len__(), desc="train", ncols=80) as t:
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
            t.update(1)



def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    inputs = []
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    with torch.no_grad():
        with tqdm(total=loader.__len__(), desc="dev", ncols=80) as t:
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(device, dtype = torch.long)
                ids = data['source_ids'].to(device, dtype = torch.long)
                mask = data['source_mask'].to(device, dtype = torch.long)

                generated_ids = model.generate(
                    input_ids = ids,
                    attention_mask = mask,
                    max_length=32,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True,
                    )
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
                input = [tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)for i in ids]

                predictions.extend(preds)
                actuals.extend(target)
                inputs.extend(input)
                for i in range(len(target)):
                    total += 1
                    preds_str = ' '.join(preds[i]).lower()
                    target_str = ' '.join(target[i]).lower()
                    if preds_str:
                      scores = rouge.get_scores(hyps=preds_str, refs=target_str)
                      rouge_1 += scores[0]['rouge-1']['f']
                      rouge_2 += scores[0]['rouge-2']['f']
                      rouge_l += scores[0]['rouge-l']['f']
                      bleu += sentence_bleu(
                          references=[target_str.split(' ')],
                          hypothesis=preds_str.split(' '),
                          smoothing_function=smooth
                      )
                t.update(1)

        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        metrics = {
              'rouge-1': rouge_1,
              'rouge-2': rouge_2,
              'rouge-l': rouge_l,
              'bleu': bleu}
    return inputs, predictions, actuals, metrics



def T5Trainer(dataframe, source_text, target_text, cfg, output_dir=None):

    torch.manual_seed(cfg.SEED)  # pytorch random seed
    np.random.seed(cfg.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    config = T5Config.from_pretrained(cfg.config_path)

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(cfg.t5_path)
    model = T5ForConditionalGeneration.from_pretrained(cfg.t5_path)
    model = model.to(cfg.device)

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]

    # Creation of Dataset and Dataloader
    train_size = 0.9
    train_dataset = dataframe.sample(frac=train_size, random_state=cfg.SEED)
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        cfg.MAX_SOURCE_TEXT_LENGTH,
        cfg.MAX_TARGET_TEXT_LENGTH,
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        cfg.MAX_SOURCE_TEXT_LENGTH,
        cfg.MAX_TARGET_TEXT_LENGTH,
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": cfg.TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": cfg.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=cfg.LEARNING_RATE
    )

    for epoch in range(cfg.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, cfg.device, training_loader, optimizer)
        inputs, predictions, actuals, metrics = validate(epoch, tokenizer, model, cfg.device, val_loader)
        print('metrics', metrics)
        # Saving the model after training
        path = os.path.join(output_dir)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)


def T5Tester(dataframe, source_text, target_text, cfg, output_dir=None):
    save_path = r'.\save_models'
    config_path = save_path + r'\config.json'
    config = T5Config.from_pretrained(config_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)
    model = T5ForConditionalGeneration.from_pretrained(save_path)
    model = model.to(cfg.device)

    test_set = YourDataSetClass(
        dataframe,
        tokenizer,
        cfg.MAX_SOURCE_TEXT_LENGTH,
        cfg.MAX_TARGET_TEXT_LENGTH,
        source_text,
        target_text
    )

    test_params = {
        "batch_size": 128,
        "shuffle": False,
        "num_workers": 0,
    }
    test_loader = DataLoader(test_set, **test_params)

    #evaluating test dataset
    for epoch in range(cfg.VAL_EPOCHS):
        inputs, predictions, actuals, metrics = validate(epoch, tokenizer, model, cfg.device, test_loader)
        print('metrics', metrics)
        final_df = pd.DataFrame({"keywords":inputs,"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_excel(os.path.join(output_dir, "pred_test.xlsx"), header=True, index=False)


def T5predicter(input):

    save_path = r'.\save_models'
    config_path = save_path + r'\config.json'
    config = T5Config.from_pretrained(config_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)
    model = T5ForConditionalGeneration.from_pretrained(save_path)
    model = model.to(cfg.device)

    input= " ".join(input.split())
    res_tokenizer = tokenizer.batch_encode_plus(
        [input],
        max_length=cfg.MAX_SOURCE_TEXT_LENGTH,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = res_tokenizer["input_ids"].to(cfg.device,dtype=torch.long)
    mask = res_tokenizer["attention_mask"].to(cfg.device,dtype=torch.long)

    #beam search
    # generated_ids = model.generate(
    #     input_ids = input_ids,
    #     attention_mask = mask ,
    #     max_length=cfg.MAX_TARGET_TEXT_LENGTH,
    #     num_beams=10,
    #     repetition_penalty=2.5,
    #     length_penalty=1.0,
    #     early_stopping=True,
    #     num_return_sequences = 5
    #     )
    #sample
    generated_ids = model.generate(
        input_ids = input_ids,
        attention_mask = mask ,
        max_length=cfg.MAX_TARGET_TEXT_LENGTH,
        do_sample=True,
        repetition_penalty=2.5,
        length_penalty=1.0,
        top_k =0,
        num_return_sequences =5
        )

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    print('input keywords:')
    print(input)
    print('generate query:')
    print(preds)





if __name__=="__main__":
    cfg = Config()
    if sys.argv[1] == 'train':
        df = load_data(r'./data/train.txt')
        T5Trainer(dataframe=df, source_text="keywords", target_text="query", cfg=cfg, output_dir="./save_models")
    elif sys.argv[1] == 'test':
        df = load_data(r'./data/test.txt')
        T5Tester(dataframe=df, source_text="keywords", target_text="query", cfg=cfg, output_dir="./outputs")
    elif sys.argv[1] == 'predict':
        T5predicter('杭州，西湖')
