import glob
import json
import re
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import dictdiffer
from tabulate import tabulate

allowed_ACT_list = ["INFORM","CONFIRM","OFFER","NOTIFY_SUCCESS","NOTIFY_FAILURE","INFORM_COUNT"]


def remove_numbers_from_string(s):
    numbers = re.findall(r'_\d+', s)
    for n in numbers:
        s = s.replace(n,"")
    s = s.lower()
    if(s=="hotels"): s = "hotel"
    if(s=="restaurants"): s = "restaurant"
    if(s=="flights"): s = "flight"
    if(s=="movies"): s = "movie"
    return s.lower()

def get_dict(DST):
    # di = {}
    di = defaultdict(lambda: defaultdict(str))
    for frame in DST:
        if(frame["state"]["active_intent"]!="NONE"):
            for k,v in frame["state"]['slot_values'].items():
                di[frame["state"]["active_intent"]][k] = v[0]
    return di

def compute_diff(DST_t,DST_t_1):
    d_t = get_dict(DST_t)
    d_t_1 = get_dict(DST_t_1)
    # print(dict(d_t_1))
    # print(dict(d_t))
    diff_list = list(dictdiffer.diff(d_t_1,d_t))
    toggle = [False,False]
    index = ["ADD","DEL"]
    for id_op, diff in enumerate(diff_list):
        if(diff[0] == "add" and diff[1] == ""):
            toggle[0] = True
            index[0] = id_op
        if(diff[0] == "remove" and diff[1] == ""):
            toggle[1] = True
            index[1] = id_op
    final_dict = defaultdict(lambda: defaultdict(str))
    if(all(toggle)):
        new_intent = diff_list[index[0]][2][0][0]
        add_dict = diff_list[index[0]][2][0][1]
        remove_dict = diff_list[index[1]][2][0][1]
        diff_slot = list(dictdiffer.diff(remove_dict,add_dict))
        for d in diff_slot:
            if(d[0]=="add"):
                for (s,v) in d[2]:
                    final_dict[new_intent][s] = v
        if(new_intent not in final_dict): final_dict[new_intent]["none"] = ""
        for id_op, diff in enumerate(diff_list):
            if(id_op not in index):
                if(diff[0] == "add"):
                    for (s,v) in diff[2]:
                        final_dict[diff[1]][s] = v
                elif(diff[0] == "change"):
                    intent_to_change, slot = diff[1].split(".")
                    final_dict[intent_to_change][slot] = diff[2][1]
    else:
        for id_op, diff in enumerate(diff_list):
            if(diff[0] == "add"):
                if(diff[1]==""):
                    for (s,v) in diff[2]:
                        final_dict[s] = v
                else:
                    for (s,v) in diff[2]:
                        final_dict[diff[1]][s] = v

            elif(diff[0] == "change"):
                intent_to_change, slot = diff[1].split(".")
                final_dict[intent_to_change][slot] = diff[2][1]

    return final_dict


def preprocessSGD_(split, develop=False):
    data = []
    files = glob.glob(f"data/dstc8-schema-guided-dialogue/{split}/*.json")
    for i_f, f in tqdm(enumerate(files),total=len(files)):
        if "schema.json" not in f:
            dialogue = json.load(open(f))
            for d in dialogue:
                # dial = {"id":d["dialogue_id"], "services": [ remove_numbers_from_string(s) for s in d["services"]],"dataset":"SGD"}
                dial = {"id":d["dialogue_id"], "services": ["SGD_"+s for s in d["services"]],"dataset":"SGD"}
                turns =[]
                dst_prev = []
                for t_idx, t in enumerate(d["turns"]):
                    if(t["speaker"]=="USER"):
                        turns.append({"dataset":"SGD","id":d["dialogue_id"],"turn_id":t_idx,"spk":t["speaker"],"utt":t["utterance"]})
                        dst_api = get_dict(t["frames"])
                        # if(len(dst_prev)==0):
                        #     dst_prev = t["frames"]
                        #     dst_api = get_dict(t["frames"])
                        # else:
                        #     dst_api = compute_diff(t["frames"],dst_prev)
                        #     dst_prev = t["frames"]
                        str_API = ''
                        serv = []
                        intent_list = set()
                        for frame in t["frames"]:
                            serv.append(remove_numbers_from_string(frame["service"]))
                            if(len(frame['state']['requested_slots'])>0):
                                for slot in frame['state']['requested_slots']:
                                    dst_api[frame['state']['active_intent']][slot] = "?"
                        for intent, slots in dst_api.items():
                            str_API += f'{intent}('
                            for s,v in slots.items():
                                if(s!='none'):
                                    str_API += f'{s}="{v.replace("(","").replace(")","")}",'
                                    intent_list.add(remove_numbers_from_string(intent))
                            if(str_API[-1]==","):
                                str_API = str_API[:-1]
                            str_API += ") "
                        # for frame in t["frames"]:
                        #     serv.append(remove_numbers_from_string(frame["service"]))
                        #     if len(frame["state"]['slot_values'])==0:
                        #         str_API += f'{frame["state"]["active_intent"]}() '

                        #     str_API += f'{frame["state"]["active_intent"]}('
                        #     for k,v in frame["state"]['slot_values'].items():
                        #         if(frame["state"]["active_intent"]!="NONE"):
                        #             v = v[0].replace('"',"'")
                        #             str_API += f'{k}="{v}",'
                        #             intent.add(remove_numbers_from_string(frame["state"]["active_intent"]))
                        #     str_API = str_API[:-1]
                        #     str_API += ") "
                        turns.append({"dataset":"SGD","id":d["dialogue_id"],"turn_id":t_idx,"spk":"API","utt":str_API,"service":list(intent_list),"service_dom":serv})
                    else:
                        group_by_act = defaultdict(list)
                        for a in t["frames"][0]["actions"]:
                            serv.append(t["frames"][0]["service"])
                            if(a["act"] in allowed_ACT_list):
                                group_by_act[a["act"]].append([a["slot"],a["values"]])
                                # str_ACT += f'{a["act"]}.{a["slot"]} = {a["values"]} '
                        str_ACT = ''
                        serv = []
                        for k,v in group_by_act.items():
                            str_ACT += f"{k}("
                            for [arg, val] in v:
                                if(len(val)>0):
                                    val = val[0].replace('"',"'")
                                    str_ACT += f'{arg}="{val}",'
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "

                        turns.append({"dataset":"SGD","id":d["dialogue_id"],"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":serv})
                        turns.append({"dataset":"SGD","id":d["dialogue_id"],"turn_id":t_idx,"spk":t["speaker"],"utt":t["utterance"]})
                dial["dialogue"] = turns
                data.append(dial)
            if(develop and i_f==1): break

    return data

def rename_service_dialogue(dial_split,name):
    new_dial = []
    for dial in dial_split:
        dial["services"] = eval(name)
        new_dial.append(dial)
    return new_dial

def preprocessSGD(develop=False):

    data = preprocessSGD_("train", develop=develop)
    data += preprocessSGD_("dev", develop=develop)
    data += preprocessSGD_("test", develop=develop)

    data_by_domain = defaultdict(list)
    for dial in data:
        if(len(dial["services"])==1):
            data_by_domain[str(sorted(dial["services"]))].append(dial)

    data_by_domain_new = defaultdict(list)
    for dom, data in data_by_domain.items():
        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.7), int(len(data)*0.8)])
        data_by_domain_new[str(sorted([remove_numbers_from_string(s) for s in eval(dom)]))].append([train_data, dev_data, test_data])

    train_data, valid_data, test_data = [], [], []
    table = []
    for dom, list_of_split in data_by_domain_new.items():
        train, valid, test = [], [], []
        for [tr,va,te] in list_of_split:
            train += rename_service_dialogue(tr,dom)
            valid += rename_service_dialogue(va,dom)
            test += rename_service_dialogue(te,dom)
        table.append({"dom":dom, "train":len(train), "valid":len(valid), "test":len(test)})
        train_data += train
        valid_data += valid
        test_data += test
    # print(tabulate(table, headers="keys"))
    # input()
    return train_data,valid_data,test_data
