import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from utils.stats import get_datasets
from collections import defaultdict

SPECIAL_TOKENS = ["[bos]", "[eos]", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '[bos]', 'eos_token': '[eos]', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer) #len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data, domains=None):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item

    def __len__(self):
        return self.dataset_len


def build_input_from_segments(args, history, plain_history, reply, row_reply,
                              tokenizer, spk, dataset, dial_id, turn_id,
                              lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """

    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    history = history[-args.max_history:]
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["padding_token"] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    instance["spk"] = spk
    instance["plain_history"] = plain_history
    instance["row_reply"] = row_reply
    instance["dataset"] = dataset
    instance["dial_id"] = dial_id
    instance["turn_id"] = turn_id
    return instance

def collate_fn(data):
    padding = data[0]["padding_token"]
    max_l = max(len(x["input_ids"]) for x in data)
    padded_dataset = {n:[] for n in MODEL_INPUTS}
    for x in data:
        padded_dataset["token_type_ids"].append( x["token_type_ids"]+ [padding]*(max_l-len(x["input_ids"])) )
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-100]*(max_l-len(x["lm_labels"]))  )
        padded_dataset["input_ids"].append(x["input_ids"]+ [padding]*(max_l-len(x["input_ids"])))
    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset

def get_pairs_from_dial(args,data,tokenizer,task_id,with_eos=True):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        d_history = []
        plain_history = []
        for idx_t, t in enumerate(dial['dialogue']):
            if(t['spk']=="USER" or t['spk']=="API-OUT"):
                utterance = tokenizer.encode(t["utt"].strip(),add_special_tokens=False)
                if(len(utterance)>200): continue
                utt_len.append(len(utterance))
                d_history.append(utterance)
                plain_history.append(t["utt"].strip())
                hist_len.append(len(sum(d_history,[])))

            elif((t['spk'] == "SYSTEM" or t['spk'] == "API") and idx_t!=0 and t["utt"]!= ""):
                utterance = tokenizer.encode(t["utt"].strip(),add_special_tokens=False)
                if(len(utterance)>200): continue

                dialogues.append(build_input_from_segments(
                                    args=args,
                                    history=list(d_history),
                                    plain_history=list(plain_history),
                                    reply=utterance if with_eos else [],
                                    row_reply=t["utt"],
                                    tokenizer=tokenizer,
                                    spk=t["spk"],
                                    dataset=t["dataset"],
                                    dial_id=t["id"],
                                    turn_id=t["turn_id"],
                                    lm_labels=True,
                                    with_eos=with_eos))
                d_history.append(utterance)
                plain_history.append(t["utt"].strip())
            else:
                utterance = tokenizer.encode(t["utt"].strip(),add_special_tokens=False)
                d_history.append(utterance)
                plain_history.append(t["utt"].strip())

    print(task_id)
    print(f"Total samples: {len(dialogues)}")
    # print(f"AVG Tok Utt: {np.mean(utt_len)}")
    # print(f"MAX Tok Utt: {max(utt_len)}")
    # print(f"MIN Tok Utt: {min(utt_len)}")
    # print(f"AVG Hist Len: {np.mean(hist_len)}")
    # print(f"MAX Hist Len: {max(hist_len)}")
    # print(f"MIN Hist Len: {min(hist_len)}")
    print()
    return dialogues

def get_data_loaders(args, tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list=args.dataset_list.split(','),setting=args.setting,verbose=args.verbose,develop=args.debug)
    if(test):
        datasets = {"test":{}}
    else:
        datasets = {"train":{}, "dev": {}, "test":{}}

    for split in datasets.keys():
        for tasks_id, task in aggregate["BYDOMAIN"][split].items():
            datasets[split][tasks_id] = get_pairs_from_dial(args,task,tokenizer,tasks_id,with_eos=False if split=="test" else True)

    task_id_train = set(task_id for task_id, dataset_task in datasets["train"].items())
    task_id_dev = set(task_id for task_id, dataset_task in datasets["dev"].items())
    task_id_test = set(task_id for task_id, dataset_task in datasets["test"].items())
    common_task_id = list(task_id_train & task_id_dev & task_id_test)
    print(common_task_id)
    train_loaders = {}
    valid_loaders = {}
    test_loaders  = {}
    if(args.continual):
        if(test):
            test_loaders = {}
            for task_id, dataset_task in datasets["test"].items():
                if(task_id in common_task_id):
                    test_loaders[task_id] = dataset_task
        else:
            for task_id, dataset_task in datasets["train"].items():
                if(task_id in common_task_id):
                    train_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.train_batch_size, shuffle=True,collate_fn=collate_fn,num_workers=40)

            for task_id, dataset_task in datasets["dev"].items():
                if(task_id in common_task_id):
                    valid_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size, shuffle=False,collate_fn=collate_fn,num_workers=40)

            for task_id, dataset_task in datasets["test"].items():
                if(task_id in common_task_id):
                    test_loaders[task_id] = dataset_task
    elif(args.multi):
        if(test):
            test_loaders = {}
            for task_id, dataset_task in datasets["test"].items():
                if(task_id in common_task_id):
                    test_loaders[task_id] = dataset_task
        else:
            dataset_train = []
            for task_id, dataset_task in datasets["train"].items():
                if(task_id in common_task_id):
                    dataset_train += dataset_task
            train_loaders = DataLoader(DatasetTrain(dataset_train), batch_size=args.train_batch_size, shuffle=True,collate_fn=collate_fn,num_workers=40)

            dataset_dev = []
            for task_id, dataset_task in datasets["dev"].items():
                if(task_id in common_task_id):
                    dataset_dev += dataset_task
            valid_loaders = DataLoader(DatasetTrain(dataset_dev), batch_size=args.valid_batch_size, shuffle=False,collate_fn=collate_fn,num_workers=40)

            for task_id, dataset_task in datasets["test"].items():
                if(task_id in common_task_id):
                    test_loaders[task_id] = dataset_task
    return train_loaders, valid_loaders, test_loaders
