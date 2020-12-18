import json
from tqdm import tqdm
from termcolor import colored

def f7(seq):
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def remove_repetition(s):
    return ".".join(f7(s.split(".")))

def get_data(dialogue,data,develop=False):
    for i_d, d in tqdm(enumerate(dialogue['dialogue_data']),total=len(dialogue['dialogue_data'])):
        dial = {"id":d["dialogue_idx"], "services":d['domains'], "dataset":"SMMID"}
        turns = []
        for t_idx, t in enumerate(d['dialogue']):
            turns.append({"dataset":"SMMID","id":d['dialogue_idx'],"turn_id":t_idx,"spk":"USER","utt":t["transcript"]})
            str_api = ""
            transcript_annotated = eval(t["transcript_annotated"])
            for trs in transcript_annotated:
                intent = trs['intent']
                # print(intent)
                # intent_prefix = ".".join(intent.split(":")[2:]).lower()
                intent = intent.split(".")[0]
                intent_prefix = "_".join(intent.split(":")).lower()
                # print(intent_prefix)
                str_api += f"{intent_prefix}("
                for s in trs['slots']:
                    if(s["id"].split(".")[0] in ["A","O",""] ):
                        slot_temp = remove_repetition(f'{s["id"].split(".")[1].replace(":","_").lower()}')
                        str_api += f'{slot_temp}="{s["text"]}",'
                    elif(s["id"].split(".")[0]=="INFO"):
                        slot_temp = remove_repetition(f'info_{s["id"].split(".")[1]},')
                        str_api += f'{slot_temp} '
                if(str_api[-1]==","):
                    str_api = str_api[:-1]
                str_api += ") "
                # if(str_api==""): str_api = intent_prefix
            turns.append({"dataset":"SMMID","id":d['dialogue_idx'],"turn_id":t_idx,"spk":"API","utt":str_api,"service":[intent_prefix.split("_")[0]]})
            str_ACT = ""
            transcript_annotated = eval(t["system_transcript_annotated"])
            for trs in transcript_annotated:
                intent = trs['intent']
                intent = intent.split(".")[0]
                intent_prefix = "_".join(intent.split(":")).lower()
                str_ACT += f"{intent_prefix}("
                for s in trs['slots']:
                    if(s["id"].split(".")[0] in ["A","O",""] ):
                        slot_temp = remove_repetition(f'{s["id"].split(".")[1].replace(":","_").lower()}')
                        str_ACT += f'{slot_temp}="{s["text"]}",'
                    elif(s["id"].split(".")[0]=="INFO"):
                        slot_temp = remove_repetition(f'info_{s["id"].split(".")[1]}')
                        str_ACT += f'{slot_temp}="{s["text"]}",'
                if(str_ACT[-1]==","):
                    str_ACT = str_ACT[:-1]
                str_ACT += ") "
                # if(str_ACT==""): str_ACT = intent_prefix

            turns.append({"dataset":"SMMID","id":d['dialogue_idx'],"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":d['domains']})
            turns.append({"dataset":"SMMID","id":d['dialogue_idx'],"turn_id":t_idx,"spk":"SYSTEM","utt":t['system_transcript']})
        dial["dialogue"] = turns
        data.append(dial)
        if(develop and i_d==10): break
    return data

def preprocessSIMMC(split,develop=False):
    data = []
    dialogue = json.load(open(f"data/simmc/data/simmc_furniture/furniture_{split}_dials.json"))
    data = get_data(dialogue,data,develop)

    dialogue = json.load(open(f"data/simmc/data/simmc_fashion/fashion_{split}_dials.json"))
    data = get_data(dialogue,data,develop)

    return data
