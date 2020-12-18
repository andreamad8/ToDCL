import re
import json
from utils.eval_metric import moses_multi_bleu
from utils.data_seq2seq import get_data_loaders
from utils.parser import get_args
from collections import defaultdict
from viz import get_viz, get_viz_folder, get_eval_from_metric
from argparse import ArgumentParser
import numpy as np
from tabulate import tabulate
from dictdiffer import diff
import glob
import os.path
from tqdm import tqdm


perm1 = {0:"['sgd_travel']",1:"['sgd_payment']",2:"['TMA_restaurant']",3:"['TMB_music']",4:"['sgd_ridesharing']",5:"['TMA_auto']",6:"['sgd_music']",7:"['sgd_buses']",8:"['TMB_restaurant']",9:"['MWOZ_attraction']",10:"['TMB_sport']",11:"['sgd_movies']",12:"['sgd_homes']",13:"['TMA_coffee']",14:"['sgd_restaurants']",15:"['sgd_hotels']",16:"['sgd_weather']",17:"['sgd_trains']",18:"['MWOZ_train']",19:"['sgd_flights']",20:"['sgd_media']",21:"['MWOZ_taxi']",22:"['sgd_alarm']",23:"['TMA_movie']",24:"['sgd_banks']",25:"['TMA_pizza']",26:"['TMB_flight']",27:"['sgd_rentalcars']",28:"['TMB_movie']",29:"['sgd_events']",30:"['MWOZ_restaurant']",31:"['sgd_services']",32:"['sgd_calendar']",33:"['TMB_food-ordering']",34:"['MWOZ_hotel']",35:"['TMA_uber']",36:"['TMB_hotel']"}
perm1 = {eval(v)[0]: k for k, v in perm1.items()}
perm2 = {0:"['sgd_trains']",1:"['TMA_restaurant']",2:"['sgd_ridesharing']",3:"['TMB_music']",4:"['sgd_alarm']",5:"['sgd_buses']",6:"['TMA_uber']",7:"['MWOZ_restaurant']",8:"['sgd_services']",9:"['sgd_rentalcars']",10:"['TMB_food-ordering']",11:"['sgd_weather']",12:"['sgd_movies']",13:"['TMA_movie']",14:"['sgd_hotels']",15:"['sgd_banks']",16:"['MWOZ_train']",17:"['MWOZ_attraction']",18:"['TMB_sport']",19:"['MWOZ_hotel']",20:"['TMB_movie']",21:"['sgd_events']",22:"['MWOZ_taxi']",23:"['sgd_travel']",24:"['sgd_music']",25:"['sgd_restaurants']",26:"['TMA_auto']",27:"['sgd_homes']",28:"['TMB_flight']",29:"['sgd_calendar']",30:"['TMB_hotel']",31:"['sgd_media']",32:"['TMB_restaurant']",33:"['sgd_flights']",34:"['sgd_payment']",35:"['TMA_coffee']",36:"['TMA_pizza']"}
perm2 = {eval(v)[0]: k for k, v in perm2.items()}
perm3 = {0:"['TMA_coffee']",1:"['TMB_music']",2:"['MWOZ_hotel']",3:"['TMA_pizza']",4:"['TMB_sport']",5:"['sgd_restaurants']",6:"['sgd_alarm']",7:"['MWOZ_train']",8:"['MWOZ_taxi']",9:"['TMA_auto']",10:"['sgd_homes']",11:"['sgd_weather']",12:"['sgd_services']",13:"['TMA_restaurant']",14:"['TMB_restaurant']",15:"['sgd_ridesharing']",16:"['sgd_calendar']",17:"['TMB_movie']",18:"['sgd_music']",19:"['sgd_events']",20:"['sgd_rentalcars']",21:"['sgd_hotels']",22:"['sgd_movies']",23:"['TMB_flight']",24:"['TMB_food-ordering']",25:"['MWOZ_attraction']",26:"['sgd_payment']",27:"['sgd_trains']",28:"['sgd_buses']",29:"['TMA_movie']",30:"['sgd_media']",31:"['TMA_uber']",32:"['sgd_banks']",33:"['sgd_flights']",34:"['TMB_hotel']",35:"['sgd_travel']",36:"['MWOZ_restaurant']"}
perm3 = {eval(v)[0]: k for k, v in perm3.items()}
perm4 = {0:"['sgd_weather']",1:"['sgd_flights']",2:"['sgd_movies']",3:"['MWOZ_taxi']",4:"['sgd_travel']",5:"['sgd_events']",6:"['sgd_trains']",7:"['sgd_hotels']",8:"['sgd_homes']",9:"['MWOZ_train']",10:"['sgd_payment']",11:"['TMA_uber']",12:"['TMB_restaurant']",13:"['sgd_services']",14:"['sgd_music']",15:"['sgd_restaurants']",16:"['TMA_pizza']",17:"['TMA_coffee']",18:"['TMB_hotel']",19:"['TMB_sport']",20:"['sgd_buses']",21:"['MWOZ_hotel']",22:"['TMB_food-ordering']",23:"['TMA_auto']",24:"['sgd_ridesharing']",25:"['sgd_calendar']",26:"['MWOZ_attraction']",27:"['TMB_movie']",28:"['TMA_movie']",29:"['sgd_alarm']",30:"['TMA_restaurant']",31:"['TMB_music']",32:"['sgd_banks']",33:"['sgd_rentalcars']",34:"['TMB_flight']",35:"['sgd_media']",36:"['MWOZ_restaurant']"}
perm4 = {eval(v)[0]: k for k, v in perm4.items()}
perm5 = {0:"['sgd_hotels']",1:"['TMB_hotel']",2:"['sgd_banks']",3:"['MWOZ_train']",4:"['sgd_music']",5:"['sgd_rentalcars']",6:"['TMB_music']",7:"['sgd_media']",8:"['TMB_restaurant']",9:"['sgd_alarm']",10:"['sgd_ridesharing']",11:"['sgd_trains']",12:"['sgd_payment']",13:"['TMA_restaurant']",14:"['TMA_uber']",15:"['MWOZ_taxi']",16:"['TMB_flight']",17:"['TMA_movie']",18:"['sgd_flights']",19:"['sgd_restaurants']",20:"['sgd_buses']",21:"['MWOZ_attraction']",22:"['TMB_movie']",23:"['TMB_food-ordering']",24:"['sgd_calendar']",25:"['TMB_sport']",26:"['TMA_pizza']",27:"['TMA_coffee']",28:"['sgd_services']",29:"['sgd_travel']",30:"['sgd_events']",31:"['MWOZ_restaurant']",32:"['sgd_homes']",33:"['TMA_auto']",34:"['sgd_weather']",35:"['sgd_movies']",36:"['MWOZ_hotel']"}
perm5 = {eval(v)[0]: k for k, v in perm5.items()}

def parse_API(text):
    API = defaultdict(lambda:defaultdict(str))
    for function in text.split(") "):
        if(function!=""):
            if("(" in function and len(function.split("("))==2):
                intent, parameters = function.split("(")
                parameters = sum([s.split('",') for s in parameters.split("=")],[])
                if len(parameters)>1:
                    if len(parameters) % 2 != 0:
                        parameters = parameters[:-1]

                    for i in range(0,len(parameters),2):
                        API[intent][parameters[i]] = parameters[i+1].replace('"',"")

                if(len(API)==0): API[intent]["none"] = "none"
    return API

def evaluate_INTENT(pred,gold,domain):
    intent_accuracy = []
    for p, g in zip(pred,gold):
        if(p.split("  ")[0].strip() == g.replace("[eos]","").strip()):
            intent_accuracy.append(1)
        else:
            intent_accuracy.append(0)
    return {"intent_accuracy":np.mean(intent_accuracy),
            "turn_level_slot_acc":0,
            "turn_level_joint_acc":0}


# def evaluate_API(pred,gold):
#     intent_accuracy = []
#     turn_level_slot_acc = []
#     turn_level_joint_acc = []
#     for p, g in zip(pred,gold):
#         API_G = {}
#         API_P = {}
#         if(g!=""):
#             API_G = parse_API(g)
#             # print(API_G)
#             if(p!="" and "(" in p and ")"): ## means the predicted text is an API
#                 API_P = parse_API(p)
#                 for k, slots in API_G.items():
#                     if(k in API_P):
#                         intent_accuracy.append(1)
#                     else:
#                         intent_accuracy.append(0)

#                     for k_p in API_P.keys():
#                         all_value = [val for _, val in API_P[k_p].items()]
#                         for s,v in slots.items():
#                             if(s in API_P[k_p]):
#                                 turn_level_slot_acc.append(1)
#                             else:
#                                 turn_level_slot_acc.append(0)

#                             if(v in all_value):
#                                 turn_level_joint_acc.append(1)
#                             else:
#                                 turn_level_joint_acc.append(0)

#             else:
#                 intent_accuracy.append(0)
#                 turn_level_joint_acc.append(0)
#                 turn_level_slot_acc.append(0)

#     return {"intent_accuracy":np.mean(intent_accuracy),
#             "turn_level_slot_acc":np.mean(turn_level_slot_acc),
#             "turn_level_joint_acc":np.mean(turn_level_joint_acc)}


def evaluate_API(pred,gold):
    intent_accuracy = []
    turn_level_slot_acc = []
    turn_level_joint_acc = []
    for p, g in zip(pred,gold):
        API_G = {}
        API_P = {}
        p = p+" "
        if(g!=""):
            API_G = parse_API(g)
            # print(API_G)
            if(p!="" and "(" in p and ")"): ## means the predicted text is an API
                API_P = parse_API(p)
                if len(API_G.keys()) != 1: 
                    continue
                if len(API_P.keys()) != 1: 
                    turn_level_joint_acc.append(0)
                    continue
                # intent accuracy
                intent_G = list(API_G.keys())[0]
                intent_P = list(API_P.keys())[0]
                if(intent_G==intent_P):
                    intent_accuracy.append(1)
                else:
                    intent_accuracy.append(0)

                state_G = {s:v for s,v in API_G[intent_G].items() if s !="none"}
                state_P = {s:v for s,v in API_P[intent_P].items() if s !="none"}
                
                if(len([d for d in diff(state_G,state_P)])==0):
                    turn_level_joint_acc.append(1)
                else:
                    turn_level_joint_acc.append(0)

            else:
                intent_accuracy.append(0)
                turn_level_joint_acc.append(0)
                turn_level_slot_acc.append(0)

    return {"intent_accuracy":np.mean(intent_accuracy),
            "turn_level_slot_acc":np.mean(turn_level_joint_acc),
            "turn_level_joint_acc":np.mean(turn_level_joint_acc)}

# ### GOOD ONE
# def evaluate_API(pred,gold):
#     intent_accuracy = []
#     turn_level_slot_acc = []
#     turn_level_joint_acc = []
#     for p, g in zip(pred,gold):
#         API_G = {}
#         API_P = {}
#         p = p+" "
#         if(g!=""):
#             API_G = parse_API(g)
#             # print(API_G)
#             if(p!="" and "(" in p and ")"): ## means the predicted text is an API
#                 API_P = parse_API(p)
#                 for k, slots in API_G.items():
#                     if(k in API_P):
#                         intent_accuracy.append(1)
#                     else:
#                         intent_accuracy.append(0)
#                     all_ = {k: True for k in API_P.keys()}
#                     if len(API_P.keys()) > 1: 
#                         turn_level_joint_acc.append(0)
#                         continue
#                         # print(g)
#                         # print(API_G)
#                         # print(p)
#                         # print(API_P)
#                         # input()
#                     for k_p in API_P.keys():
#                         for s,v in slots.items():
#                             if(s =="none"): 
#                                 turn_level_slot_acc.append(1)
#                                 break
#                             if(API_P[k_p][s] == v):
#                                 turn_level_slot_acc.append(1)
#                             else:
#                                 turn_level_slot_acc.append(0)
#                                 all_[k_p] = False
#                         if(all_[k_p]):
#                             turn_level_joint_acc.append(1)
#                         else:
#                             turn_level_joint_acc.append(0)
#                         break

#             else:
#                 intent_accuracy.append(0)
#                 turn_level_joint_acc.append(0)
#                 turn_level_slot_acc.append(0)
#         # print({"intent_accuracy":np.mean(intent_accuracy),
#         #     "turn_level_slot_acc":np.mean(turn_level_slot_acc),
#         #     "turn_level_joint_acc":np.mean(turn_level_joint_acc)})
            
#     return {"intent_accuracy":np.mean(intent_accuracy),
#             "turn_level_slot_acc":np.mean(turn_level_slot_acc),
#             "turn_level_joint_acc":np.mean(turn_level_joint_acc)}





# def evaluate_API(pred,gold):
#     intent_accuracy = []
#     turn_level_slot_acc = []
#     turn_level_joint_acc = []
#     for p, g in zip(pred,gold):
#         API_G = {}
#         API_P = {}
#         p = p+" "
#         if(g!=""):
#             API_G = parse_API(g)
#             # print(API_G)
#             if(p!="" and "(" in p and ")"): ## means the predicted text is an API
#                 API_P = parse_API(p)
#                 print(g)
#                 print(API_G)
#                 print(p)
#                 print(API_P)
#                 input()
#                 for k, slots in API_G.items():
#                     if(k in API_P):
#                         intent_accuracy.append(1)
#                     else:
#                         intent_accuracy.append(0)
#                     all_ = {k: True for k in API_P.keys()}
#                     # assert len(API_P.keys()) <= 1
#                     for k_p in API_P.keys():
#                         for s,v in slots.items():
#                             if(API_P[k_p][s] == v):
#                                 turn_level_slot_acc.append(1)
#                             else:
#                                 turn_level_slot_acc.append(0)
#                                 all_[k_p] = False
#                         if(all_[k_p]):
#                             turn_level_joint_acc.append(1)
#                         else:
#                             turn_level_joint_acc.append(0)

#             else:
#                 intent_accuracy.append(0)
#                 turn_level_joint_acc.append(0)
#                 turn_level_slot_acc.append(0)
#         print({"intent_accuracy":np.mean(intent_accuracy),
#             "turn_level_slot_acc":np.mean(turn_level_slot_acc),
#             "turn_level_joint_acc":np.mean(turn_level_joint_acc)})
            
#     return {"intent_accuracy":np.mean(intent_accuracy),
#             "turn_level_slot_acc":np.mean(turn_level_slot_acc),
#             "turn_level_joint_acc":np.mean(turn_level_joint_acc)}


def evaluate_EER(args,results_dict,entities_json,path, names):
    ERR = []
    cnt_bad = 0
    cnt_superflous = 0
    tot = 0
    
    for d in results_dict:
        if(d["spk"]=="SYSTEM"):
            ent = set()
            ent_corr = []
            if args.task_type == "E2E":
                d['hist'] = d['hist'].split("API-OUT: ")[1]
                if(d['hist']==""):
                    continue

            for speech_act, slot_value_dict in parse_API(d['hist']+" ").items():
                tot += len(slot_value_dict.keys())
                for s,v in slot_value_dict.items():
                    if(v not in ["True", "False", "yes", "no", "?","none"]):
                        if(v.lower() not in d["genr"].lower()):
                            cnt_bad += 1
                        else:
                            ent_corr.append(v.lower())
                        ent.add(v.lower())
                    
                # temp = str(d["genr"].lower())
                # for e in sorted(list(set(ent_corr)), key=len, reverse=True):
                #     temp = temp.replace(e,"")

                # for slot, values in entities_json[d['task_id']].items():
                #     for v in values-ent:

                #         if(v.lower() in temp):
                #             cnt_superflous += 1 
                #             print(d['hist'])
                #             print(d["gold"])
                #             print(v)
                #             input()
                #             break

    return (cnt_bad+cnt_superflous)/float(tot)
    # return (cnt_bad)/float(tot)
                        # else:
                        #     ent_corr.append(v)
                        # ent.add(v)
            # for e in sorted(list(set(ent_corr)), key=len, reverse=True):
            #     temp = d["genr"].replace(e,"")
            # superflous = []
            # for slot, values in entities_json[d['task_id']].items():
            #     for v in set(values)-ent:
            #         if(v in [str(i) for i in range(100)]):
            #             continue
            #         if(v in temp):
            #             superflous.append(v)
            #             # print(d['hist'])
            #             # print(d["gold"])
            #             # print(v)
            #             # input()
            #             break
            # q = 0 #len(set(superflous))
            # # print(superflous)
            # if(M!=0):
            #     tot += 1
            #     ERR.append((p+q)/M)
                # if args.all:
                #     ts_id_gold = names.index(eval(d['task_id'])[0])
                # else:
                #     ts_id_gold = names.index(eval(allign_to_name(d['task_id']))[0])
                # if((p+q)/M != 0 and "ADAPTER" in path):
                #     if(ts_id_gold != d['pred_task_id']):
                #         cnt_bad += 1
                      
                        # print(d['hist'])
                        # print(parse_API(d['hist']+" "))
                        # print(M)
                        # print(d["gold"])
                        # print(d["genr"])
                        # print(p)
                        # print(q)
                        # print((p+q)/M)
                        # print("RIGHT:",d['task_id'])
                        # print("PREDI:",d['pred_task_id'])
                        # input()


    # return np.mean(ERR)




def evaluate(args,path,names,ent):
    results_json = json.load(open(path))
    entities_json = ent
    # if args.all:
    #     entities_json = json.load(open("data/entities_SGD,TM19,TM20,MWOZ.json"))
    # else:
    #     entities_json = json.load(open("data/entities_SGD.json"))
    acc = 0
    if("ADAPTER" in path):
        acc = []
        for r in results_json:
            if args.all:
                ts_id_gold = names.index(eval(r['task_id'])[0])
            else:
                ts_id_gold = names.index(eval(allign_to_name(r['task_id']))[0])
            if(ts_id_gold == r['pred_task_id']):
                acc.append(1)
            else:
                acc.append(0)
        acc = np.mean(acc)
        # print("ACC:",np.mean(acc))



    domain_BLEU = defaultdict(lambda: defaultdict(list))
    domain_API = defaultdict(lambda: defaultdict(list))
    domain_NLG = defaultdict(list)
    for r in results_json:
        if(r['spk']=='SYSTEM'):
            domain_BLEU[r['task_id']]["pred"].append(r['genr'].strip())
            domain_BLEU[r['task_id']]["gold"].append(r['gold'].replace("[eos]","").strip())
            domain_NLG[r['task_id']].append(r)
        elif(r['spk']=='API'):
            domain_API[r['task_id']]["pred"].append(r['genr'])
            domain_API[r['task_id']]["gold"].append(r['gold'])

    T_BLEU = {}
    T_NLG = {}
    if args.task_type =="NLG" or args.task_type =="E2E":
        for k, sample_NLG in domain_NLG.items():
            T_NLG[k] = evaluate_EER(args,sample_NLG,entities_json, path, names)
        for k,v in domain_BLEU.items():
            T_BLEU[k] = moses_multi_bleu(v["pred"],v["gold"])

    T_API = {}
    for k,v in domain_API.items():
        if args.task_type =="NLG":
            T_API[k] = 0
        if args.task_type =="INTENT":
            # if("['sgd_travel']" in path and k=="['sgd_travel']"):
            #     T_API[k] = evaluate_INTENT(v["pred"],v["gold"],domain=k)
            # else:
            T_API[k] = evaluate_INTENT(v["pred"],v["gold"],domain="")
        if args.task_type =="E2E" or args.task_type =="DST":
            T_API[k] = evaluate_API(v["pred"],v["gold"])

    return {"API":["G_API","D_API",T_API],"BLEU":["G_BLEU","D_BLEU",T_BLEU], "EER":T_NLG, "ACC":acc}


def allign_to_name(k):
    k = k.replace("sgd_","")
    if(k =="['restaurants']"):
        k = "['restaurant']"
    if(k =="['hotels']"):
        k = "['hotel']"
    if(k =="['flights']"):
        k = "['flight']"
    if(k =="['movies']"):
        k = "['movie']"
    return k

def score_folder():

    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path to the folder with the results")
    parser.add_argument("--task_type", type=str, default="E2E", help="Path to the folder with the results")
    parser.add_argument("--best", action='store_true', help="best model only")
    parser.add_argument("--all", action='store_true', help="all dataset")
    parser.add_argument("--ablation", action='store_true', help="all dataset")
    parser.add_argument("--adapter", action='store_true', help="all dataset")
    
    args = parser.parse_args()
    folders = glob.glob(f"{args.model_checkpoint}/*")

    entities_j = json.load(open("data/entities_SGD,TM19,TM20,MWOZ.json"))
    integers = [str(i) for i in range(100)]
    entities_json = defaultdict(lambda: defaultdict(set))
    for k,v in entities_j.items():
        for slot, values in v.items():
            entities_json[k][slot] = set([val.lower() for val in values if v not in integers])
    ##
    # table_API = []
    # table_BLEU = []
    # table_EER = []
    # table_VALUE = []
    # for permu in [1,2,3,4,5]:
    #     if(permu==1):names = list(perm1.keys())
    #     if(permu==2):names = list(perm2.keys())
    #     if(permu==3):names = list(perm3.keys())
    #     if(permu==4):names = list(perm4.keys())
    #     if(permu==5):names = list(perm5.keys())
    #     direct = f'runs_{args.task_type}/BEST/ADAPTER_EPC_10_LR_0.00625_BOTL_300_PERM_{permu}_gpt2'
    #     results_multitask = evaluate(args,f'{direct}/gold_id_response_perm_{permu}.json',names)
    #     A_row_BLEU = ["" for _ in range(len(names))]
    #     for k,v in results_multitask["BLEU"][2].items():
    #         A_row_BLEU[names.index(eval(k)[0])] = v

    #     A_row_EER = ["" for _ in range(len(names))]
    #     for k,v in results_multitask["EER"].items():
    #         A_row_EER[names.index(eval(k)[0])] = v

    #     A_row_API = ["" for _ in range(len(names))]
    #     for k,v in results_multitask["API"][2].items():
    #         A_row_API[names.index(eval(k)[0])] = v["intent_accuracy"]

    #     A_row_SLOT = ["" for _ in range(len(names))]
    #     for k,v in results_multitask["API"][2].items():
    #         A_row_SLOT[names.index(eval(k)[0])] = v["turn_level_slot_acc"]

    #     A_row_VALUE = ["" for _ in range(len(names))]
    #     for k,v in results_multitask["API"][2].items():
    #         A_row_VALUE[names.index(eval(k)[0])] = v["turn_level_joint_acc"]
    #     if(args.task_type=="INTENT" or args.task_type=="E2E"):
    #         table_API.append({"Method":"ADPT_GOLD","ACC":np.mean(A_row_API),"BWT":0,"FWT":0})
    #     if(args.task_type=="NLG" or args.task_type=="E2E"):
    #         table_BLEU.append({"Method":"ADPT_GOLD","ACC":np.mean(A_row_BLEU),"BWT":0,"FWT":0})
    #         table_EER.append({"Method":"ADPT_GOLD","ACC":np.mean(A_row_EER),"BWT":0,"FWT":0})
    #     if(args.task_type=="DST" or args.task_type=="E2E"):
    #         table_VALUE.append({"Method":"ADPT_GOLD","ACC":np.mean(A_row_VALUE),"BWT":0,"FWT":0})
    # print("INTENT")
    # print(tabulate(table_API, headers="keys"))
    # print("NLG")
    # print(tabulate(table_BLEU, headers="keys"))
    # print(tabulate(table_EER, headers="keys"))
    # print("DST")
    # print(tabulate(table_VALUE, headers="keys"))

    # exit()

    ## this just to get the task_id name
    # for folder in folders:
    #     if "VANILLA" in folder:
    #         files = glob.glob(f"{folder}/*")
    #         names = []
    #         for f in files:
    #             if "png" not in f and "json" not in f and "bin" not in f and "txt" not in f and ".model" not in f:
    #                 if(args.all):
    #                     task_id = eval("_".join(f.split("/")[-1].split("_")[1:]))
    #                 else:
    #                     task_id = eval(f.split("/")[-1].split("_")[1])
    #                 if(task_id[0]!=""): 
    #                     names.append(task_id[0])
    #         break
    # print(folders)
    names = list(perm5.keys())
    list_matrix_BLEU = []
    list_b_BLEU = []
    list_matrix_EER = []
    list_b_EER = []
    list_matrix_INTENT = []
    list_b_INTENT = []
    list_matrix_SLOT = []
    list_b_SLOT = []
    list_matrix_VALUE = []
    list_b_VALUE = []
    list_multi_BLEU = []
    list_multi_EER = []
    list_multi_INTENT = []
    list_multi_SLOT = []
    list_multi_VALUE = []
    methods_name = []
    methods_name_multi = []
    acc_tot = []
    for folder in folders:
        if "png" in folder or "TOO_HIGH_LR" in folder or "TEMP" in folder:
            continue
        if ("MULTI" not in folder):
            files = glob.glob(f"{folder}/*")

            if(args.best):
                methods_name.append(folder.split("/")[-1].split("_")[0])
            else:
                methods_name.append(folder.split("/")[-1])#.split("_")[0])
            matrix_results_BLEU = []
            matrix_results_ERR = []
            matrix_results_INTENT = []
            matrix_results_SLOT = []
            matrix_results_VALUE = []
            for f in files:
                if "png" not in f and "json" not in f and "bin" not in f and "txt" not in f and ".model" not in f:
                    print(f)
                    if args.all:
                        task_id = eval("_".join(f.split("/")[-1].split("_")[1:]))
                    else:
                        task_id = eval(f.split("/")[-1].replace("sgd_","").split("_")[1])
                    if(os.path.isfile(f+'/generated_responses.json')):
                        if("PERM_1" in f):names = list(perm1.keys())
                        if("PERM_2" in f):names = list(perm2.keys())
                        if("PERM_3" in f):names = list(perm3.keys())
                        if("PERM_4" in f):names = list(perm4.keys())
                        if("PERM_5" in f):names = list(perm5.keys())
                        # else: names = list(perm5.keys())
                        results = evaluate(args,f+'/generated_responses.json',names, ent=entities_json)
                        if("ADAPTER" in f and "36" in f):
                            acc_tot.append(results["ACC"])

                        if args.task_type =="E2E" or args.task_type =="NLG":
                            row = ["" for _ in range(len(names))]
                            for k,v in results["BLEU"][2].items():
                                if(not args.all):
                                    k = allign_to_name(k)
                                row[names.index(eval(k)[0])] = v
                            matrix_results_BLEU.append(row)

                            row = ["" for _ in range(len(names))]
                            for k,v in results["EER"].items():
                                if(not args.all):
                                    k = allign_to_name(k)
                                row[names.index(eval(k)[0])] = v
                            matrix_results_ERR.append(row)
                        else: 
                            matrix_results_BLEU.append([0.0 for _ in range(len(names))])
                            matrix_results_ERR.append([0.0 for _ in range(len(names))])

                        if args.task_type =="E2E" or args.task_type =="INTENT":
                            row = ["" for _ in range(len(names))]
                            for k,v in results["API"][2].items():
                                if(not args.all):
                                    k = allign_to_name(k)
                                row[names.index(eval(k)[0])] = v["intent_accuracy"]
                            matrix_results_INTENT.append(row)
                        else:
                            matrix_results_INTENT.append([0.0 for _ in range(len(names))])

                        if args.task_type =="E2E" or args.task_type =="DST":
                            row = ["" for _ in range(len(names))]
                            for k,v in results["API"][2].items():
                                if(not args.all):
                                    k = allign_to_name(k)
                                row[names.index(eval(k)[0])] = v["turn_level_slot_acc"]
                            matrix_results_SLOT.append(row)

                            row = ["" for _ in range(len(names))]
                            for k,v in results["API"][2].items():
                                if(not args.all):
                                    k = allign_to_name(k)
                                row[names.index(eval(k)[0])] = v["turn_level_joint_acc"]
                            matrix_results_VALUE.append(row)
                        else:
                            matrix_results_SLOT.append([0.0 for _ in range(len(names))])
                            matrix_results_VALUE.append([0.0 for _ in range(len(names))])
                    else:
                        matrix_results_BLEU.append([0.0 for _ in range(len(names))])
                        matrix_results_INTENT.append([0.0 for _ in range(len(names))])
                        matrix_results_SLOT.append([0.0 for _ in range(len(names))])
                        matrix_results_VALUE.append([0.0 for _ in range(len(names))])
                        matrix_results_ERR.append([0.0 for _ in range(len(names))])

            list_matrix_BLEU.append(matrix_results_BLEU)
            list_b_BLEU.append([0 for r in matrix_results_BLEU])

            list_matrix_EER.append(matrix_results_ERR)
            list_b_EER.append([0 for r in matrix_results_ERR])
            list_matrix_INTENT.append(matrix_results_INTENT)
            list_b_INTENT.append([0 for r in matrix_results_BLEU])
            list_matrix_SLOT.append(matrix_results_SLOT)
            list_b_SLOT.append([0 for r in matrix_results_BLEU])
            list_matrix_VALUE.append(matrix_results_VALUE)
            list_b_VALUE.append([0 for r in matrix_results_BLEU])
        else:
            results_multitask = evaluate(args,f'{folder}/multi/generated_responses.json',names, ent=entities_json)
            m_row_BLEU = ["" for _ in range(len(names))]
            for k,v in results_multitask["BLEU"][2].items():
                m_row_BLEU[names.index(eval(k)[0])] = v

            m_row_EER = ["" for _ in range(len(names))]
            for k,v in results_multitask["EER"].items():
                m_row_EER[names.index(eval(k)[0])] = v


            m_row_API = ["" for _ in range(len(names))]
            for k,v in results_multitask["API"][2].items():
                m_row_API[names.index(eval(k)[0])] = v["intent_accuracy"]

            m_row_SLOT = ["" for _ in range(len(names))]
            for k,v in results_multitask["API"][2].items():
                m_row_SLOT[names.index(eval(k)[0])] = v["turn_level_slot_acc"]

            m_row_VALUE = ["" for _ in range(len(names))]
            for k,v in results_multitask["API"][2].items():
                m_row_VALUE[names.index(eval(k)[0])] = v["turn_level_joint_acc"]

            list_multi_BLEU.append(m_row_BLEU)
            list_multi_EER.append(m_row_EER)
            list_multi_INTENT.append(m_row_API)
            list_multi_SLOT.append(m_row_SLOT)
            list_multi_VALUE.append(m_row_VALUE)
            methods_name_multi.append(folder.split("/")[-1].split("_")[0])


    if args.task_type =="E2E" or args.task_type =="NLG":
        get_viz_folder(args,list_matrix_BLEU,list_b_BLEU,names,list_multi_BLEU,methods_name,methods_name_multi,"BLEU")
        get_viz_folder(args,list_matrix_EER,list_b_EER,names,list_multi_EER,methods_name,methods_name_multi,"EER")
    if args.task_type =="E2E" or args.task_type =="INTENT":
        get_viz_folder(args,list_matrix_INTENT,list_b_INTENT,names,list_multi_INTENT,methods_name,methods_name_multi,"INTENT")
    if args.task_type =="E2E" or args.task_type =="DST":
        get_viz_folder(args,list_matrix_SLOT,list_b_SLOT,names,list_multi_SLOT,methods_name,methods_name_multi,"DST-SLOT")
        get_viz_folder(args,list_matrix_VALUE,list_b_VALUE,names,list_multi_VALUE,methods_name,methods_name_multi,"DST-VALUE")
    print("ADAPTER MEAN: ", np.mean(acc_tot))
    print("ADAPTER STD: ",np.std(acc_tot))
# if arg.folder:
score_folder()
# else:
#     score_single()

# from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel,T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
# import random
# from pytorch_lightning import Trainer, seed_everything

# args = get_args()
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token="[bos]", eos_token="[eos]", sos_token="[SOS]", sep_token="[sep]",pad_token='[PAD]')
# args.dataset_list = "SGD"
# args.task_type = "NLG"
# args.continual = True
# train_loader, val_loader, dev_val_loader, (train_datasets, test_datasets) = get_data_loaders(args,tokenizer)

# keys =  list(train_loader.keys())
# seed_everything(1)
# random.shuffle(keys)
# new_load = {key: train_loader[key] for key in keys}
# for k,_ in new_load.items():
#     print(k)
# print()
# seed_everything(2)
# random.shuffle(keys)
# new_load = {key: train_loader[key] for key in keys}
# for k,_ in new_load.items():
#     print(k)
# print()
# seed_everything(3)
# random.shuffle(keys)
# new_load = {key: train_loader[key] for key in keys}
# for k,_ in new_load.items():
#     print(k)
# value_by_dom_by_slots = defaultdict(lambda:defaultdict(set))
# for dom, data in train_datasets.items():
#     for d in data:
#         for speech_act, slot_value_dict in parse_API(d['history']+" ").items():
#             for s,v in slot_value_dict.items():
#                 if(v not in ["True", "False", "yes", "no", "?","none"]):
#                     value_by_dom_by_slots[dom][s].add(v)

# value_by_dom_by_slots_ = defaultdict(lambda:defaultdict(list))

# for dom, slotval in value_by_dom_by_slots.items():
#     # print(f"DOMAIN: {dom}")
#     for s,v in slotval.items():
#         value_by_dom_by_slots_[allign_to_name(dom)][s] = list(v)


# with open(f'entities_SGD.json', 'w') as fp:
#     json.dump(value_by_dom_by_slots_, fp, indent=4)
