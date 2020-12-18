from termcolor import colored
import glob
import json
from collections import defaultdict
from tqdm import tqdm

def get_value_dst(DST):
    active_dst = defaultdict(list)
    for k,v in DST.items():
        for k_s, v_s in v['semi'].items():
            if(len(v_s)!=0):
                active_dst[k].append([k_s, v_s])
        for k_s, v_s in v['book'].items():
            if(len(v_s)!=0 and k_s != "booked"):
                active_dst[k].append([k_s, v_s])
    return active_dst

def compute_diff(DST_t,DST_t_1):
    active_dst = defaultdict(list)
    for (k_t,v_t), (k_t_1,v_t_1) in zip(DST_t.items(),DST_t_1.items()):
        for (k_t_s,v_t_s), (k_t_1_s,v_t_1_s) in zip(v_t['semi'].items(),v_t_1['semi'].items()):
            if(v_t_s != v_t_1_s):
                active_dst[k_t].append([k_t_s, v_t_s])
        for (k_t_s,v_t_s), (k_t_1_s,v_t_1_s) in zip(v_t['book'].items(),v_t_1['book'].items()):
            if(v_t_s != v_t_1_s and k_t_s != "booked"):
                active_dst[k_t].append([k_t_s, v_t_s])
    return active_dst

def get_domains(goal):
    dom = []
    for d, g in goal.items():
        if(len(g)!=0) and d!= "message" and d!= "topic":
            dom.append("MWOZ_"+d)
    return dom

def loadCSV(split):
    split_id = []
    with open(f"data/multiwoz/data/MultiWOZ_2.1/{split}ListFile.txt") as f:
        for l in f:
            split_id.append(l.replace("\n",""))
    return split_id

def preprocessMWOZ(develop=False):
    data = []
    dialogue = json.load(open("data/multiwoz/data/MultiWOZ_2.2/data.json"))
    for i_d, (d_idx, d) in tqdm(enumerate(dialogue.items()),total=len(dialogue.items())):
        dial = {"id":d_idx, "services": get_domains(d['goal']), "dataset":"MWOZ"}
        if "MWOZ_police" in dial["services"] or "MWOZ_hospital" in dial["services"] or "MWOZ_bus" in dial["services"]: continue
        turns =[]
        dst_prev = {}
        for t_idx, t in enumerate(d['log']):
            if(t_idx % 2 ==0):
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":t["text"]})
                # print("USER",t["text"])
                str_API_ACT = ""
                if "dialog_act" in t:
                    intents_act = set()
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_API_ACT += f'{s.lower()}="{v}",'
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
            else:
                dst_api = get_value_dst(t["metadata"])

                # if(len(dst_prev)==0):
                #     dst_prev = t["metadata"]
                #     dst_api = get_value_dst(t["metadata"])
                # else:
                #     dst_api = compute_diff(t["metadata"],dst_prev)
                #     dst_prev = t["metadata"]

                str_API = ""
                intents = set()
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            str_API += f'{s.lower()}="{v}",'
                            intents.add(k.lower().replace('-','_'))
                    if(len(str_API)>0 and str_API[-1]==","):
                        str_API = str_API[:-1]
                    str_API += ") "
                if(str_API==""):
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API_ACT,"service":list(intents_act)})
                    # print("API",str_API_ACT)
                else:
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API,"service":list(intents)})
                    # print("API", str_API)

                ## API RETURN
                str_ACT = ""
                if "dialog_act" in t:
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_ACT += f'{s.lower()}="{v}",'
                                    # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})
        dial["dialogue"] = turns
        data.append(dial)
        if(develop and i_d==10): break


    split_id_dev, split_id_test = loadCSV("val"), loadCSV("test")

    train_data, dev_data, test_data = [], [], []

    for dial in data:
        if dial["id"] in split_id_dev:
           dev_data.append(dial)
        elif dial["id"] in split_id_test:
           test_data.append(dial)
        else:
           train_data.append(dial)
    return train_data, dev_data, test_data
