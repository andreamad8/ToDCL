import glob
import json
import re
import numpy as np
from tqdm import tqdm
from termcolor import colored

def _preprocessMETA(f,develop=False):

    data = []

    dialogue = open(f,'r')
    for i_d, d in enumerate(dialogue):
        d = eval(d)
        dial = {"id":d["id"], "services": ["METAL_"+d["domain"].lower()],"dataset":"METAL"}
        turns =[]
        for t_idx, t in enumerate(d['turns']):
            if(t_idx % 2 ==0):
                turns.append({"dataset":"METAL","id":d["id"],"turn_id":t_idx,"spk":"API-OUT","utt":""})
                turns.append({"dataset":"METAL","id":d["id"],"turn_id":t_idx,"spk":"SYSTEM","utt":t})
            else:
                turns.append({"dataset":"METAL","id":d["id"],"turn_id":t_idx,"spk":"USER","utt":t})
                turns.append({"dataset":"METAL","id":d["id"],"turn_id":t_idx,"spk":"API","utt":"","service":""})
        dial["dialogue"] = turns[2:]
        data.append(dial)
        if(develop and i_d==10): break


    train_data, dev_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
    return train_data.tolist(), dev_data.tolist(), test_data.tolist()

def preprocessMETA(develop=False):
    train, dev, test = [], [], []
    files = glob.glob(f"data/metalwoz-v1/dialogues/*.txt")
    for f in tqdm(files,total=len(files)):
        train_, dev_, test_ = _preprocessMETA(f,develop=develop)
        train += train_
        dev += dev_
        test += test_
    return train, dev, test
