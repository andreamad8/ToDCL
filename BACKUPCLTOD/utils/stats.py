from data.MWOZ import preprocessMWOZ
from data.SGD import preprocessSGD
from data.SMMID import preprocessSIMMC
from data.TM import preprocessTM2019,preprocessTM2020
from data.METAL import preprocessMETA
from termcolor import colored
import numpy as np
import random
import re
from tabulate import tabulate
from termcolor import colored
from collections import defaultdict


def get_domains_slots(data):
    services = set()
    intent = set()
    len_dialogue = []
    for dial in data:
        for s in dial["services"]:
            services.add(s)
        len_dialogue.append(len([0 for t in dial['dialogue'] if t["spk"] in ["USER","SYSTEM"]]))
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if(" " in s or len(s)==1):
                        print(s) 
                        print(turn)
                        input()
                    intent.add(s)
    print("Domain",len(services))
    print("Intent",len(intent))
    print("Avg. Turns",np.mean(len_dialogue))
    return len(services), len(intent), np.mean(len_dialogue), intent

def filter_services(data,serv):
    filtered_dialogue = []
    for dial in data:
        flag_temp = True
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if s not in serv:
                        flag_temp = False
        if(flag_temp):
            filtered_dialogue.append(dial)
    return filtered_dialogue


def split_by_domain(data,setting):
    data_by_domain = defaultdict(list)
    if setting=="single":
        for dial in data:
            if(len(dial["services"])==1):
                data_by_domain[str(sorted(dial["services"]))].append(dial)
        print("SINGLE DOMAIN:",len(data_by_domain.keys()))

    elif setting=="multi":
        data_by_domain = defaultdict(list)
        for dial in data:
            data_by_domain[str(sorted(dial["services"]))].append(dial)
        print("ALL DOMAIN:",len(data_by_domain.keys()))
    else:
        print("choose a setting: --setting single or --setting multi")
        exit(1)
    # for d,v in sorted(data_by_domain.items() ,  key=lambda x: len (eval(x[0]))):
    #     print(d)
    return dict(sorted(data_by_domain.items() ,  key=lambda x: len (eval(x[0]))))


def print_sample(data,num):
    color_map = {"USER":"blue","SYSTEM":"magenta","API":"red","API-OUT":"green"}
    for i_d, dial in enumerate(random.sample(data,len(data))):
        print(f'ID:{dial["id"]}')
        print(f'Services:{dial["services"]}')
        for turn in dial['dialogue']:
            print(colored(f'{turn["spk"]}:',color_map[turn["spk"]])+f' {turn["utt"]}')
        if i_d == num: break

def get_datasets(dataset_list=['SGD'],setting="single",verbose=False,develop=False):

    table = []
    train = []
    dev = []
    test = []
    if("TM19" in dataset_list):
        print("LOAD TM19")
        train_TM19, dev_TM19, test_TM19 = preprocessTM2019(develop=develop)
        if(verbose):
            print_sample(train_TM19,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_TM19)
        table.append({"Name":"TM19","Trn":len(train_TM19),"Val":len(dev_TM19),"Tst":len(test_TM19),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_TM19
        dev += dev_TM19
        test += test_TM19

    if("TM20" in dataset_list):
        print("LOAD TM20")
        train_TM20, dev_TM20, test_TM20 = preprocessTM2020(develop=develop)
        if(verbose):
            print_sample(train_TM20,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_TM20)
        table.append({"Name":"TM20","Trn":len(train_TM20),"Val":len(dev_TM20),"Tst":len(test_TM20),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_TM20
        dev += dev_TM20
        test += test_TM20

    if("MWOZ" in dataset_list):
        print("LOAD MWOZ")
        train_MWOZ, dev_MWOZ,test_MWOZ = preprocessMWOZ(develop=develop)
        if(verbose):
            print_sample(train_MWOZ,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_MWOZ)
        table.append({"Name":"MWOZ","Trn":len(train_MWOZ),"Val":len(dev_MWOZ),"Tst":len(test_MWOZ),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_MWOZ
        dev += dev_MWOZ
        test += test_MWOZ

    if("SGD" in dataset_list):
        print("LOAD SGD")
        train_SGD, dev_SGD, test_SGD = preprocessSGD(develop=develop)
        if(verbose):
            print_sample(train_SGD,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_SGD)
        table.append({"Name":"SGD","Trn":len(train_SGD),"Val":len(dev_SGD),"Tst":len(test_SGD),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_SGD
        dev += dev_SGD
        test += test_SGD

    if("SMMIC" in dataset_list):
        print("LOAD SMMIC")
        train_SMMIC = preprocessSIMMC("train",develop=develop)
        dev_SMMIC = preprocessSIMMC("dev",develop=develop)
        test_SMMIC = preprocessSIMMC("devtest",develop=develop)
        if(verbose):
            print_sample(train_SMMIC,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_SMMIC)
        table.append({"Name":"SMMIC","Trn":len(train_SMMIC),"Val":len(dev_SMMIC),"Tst":len(test_SMMIC),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_SMMIC
        dev += dev_SMMIC
        test += test_SMMIC

    if("METAL" in dataset_list):
        print("LOAD METALWOZ")
        train_METAL, dev_METAL, test_METAL = preprocessMETA(develop=develop)
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_METAL)
        if(verbose):
            print_sample(train_METAL,2)
            input()
        table.append({"Name":"METAL","Trn":len(train_METAL),"Val":len(dev_METAL),"Tst":len(test_METAL),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_METAL
        dev += dev_METAL
        test += test_METAL


    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
    test = filter_services(test,services) ## Remove dialogue with API not present in the test set
    dev = filter_services(dev,services) ## Remove dialogue with API not present in the test set
    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    if(verbose):
        for inten in services:
            print(inten)
        input()
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
    print(tabulate(table, headers="keys"))

    return {"TOTAL":{"train":train,"dev":dev,"test":test},
               "BYDOMAIN":{"train":split_by_domain(train,setting),
                            "dev":split_by_domain(dev,setting),
                            "test":split_by_domain(test,setting)}
                            }

# {"TM19":{"train":train_TM19,"dev":dev_TM19,"test":test_TM19},
#             "TM20":{"train":train_TM20,"dev":dev_TM20,"test":test_TM20},
#             "MWOZ":{"train":train_MWOZ,"dev":dev_MWOZ,"test":test_MWOZ},
#             "SGD":{"train":train_SGD,"dev":dev_SGD,"test":test_SGD},
#             "SMMIC":{"train":train_SMMIC,"dev":dev_SMMIC,"test":test_SMMIC},
#             "METALWOZ":{"train":train_METAL,"dev":dev_METAL,"test":test_METAL},
#             },
