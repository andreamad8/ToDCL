from train_CL import Seq2SeqToD
import torch
from utils.parser import get_args
from utils.data_seq2seq import get_data_loaders,make_loader
from test import test_generation_GPT2BATCH
from collections import defaultdict
from tqdm import tqdm
import json


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

# permu = 1
task = "E2E"
for permu in [1,2,3,4,5]:
    direct = f'runs_{task}/BEST/ADAPTER_EPC_10_LR_0.00625_BOTL_300_PERM_{permu}_gpt2'
    print(direct)
    args = get_args()
    args.number_of_adpt = 40
    args.bottleneck_size = 300
    args.task_type = task
    args.CL = "ADAPTER"
    args.dataset_list = "SGD,TM19,TM20,MWOZ"
    args.model_checkpoint = "gpt2"
    model = Seq2SeqToD(args)
    state_dict = torch.load(f'{direct}/pytorch_model.bin')
    model.model.load_state_dict(state_dict)
    train_loader, val_loader, dev_val_loader, (train_datasets, val_datasets, test_datasets) = get_data_loaders(args, model.tokenizer)
    for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
        model.task_list_seen.append(task_id)

    device = torch.device(f"cuda:0")
    model.model.to(device)
    model.model.eval()
    results = []


    test_by_task_id = defaultdict(list)
    for sample in test_datasets:
        if(permu == 1): test_by_task_id[perm1[eval(sample["task_id"])[0]]].append(sample)
        if(permu == 2): test_by_task_id[perm2[eval(sample["task_id"])[0]]].append(sample)
        if(permu == 3): test_by_task_id[perm3[eval(sample["task_id"])[0]]].append(sample)
        if(permu == 4): test_by_task_id[perm4[eval(sample["task_id"])[0]]].append(sample)
        if(permu == 5): test_by_task_id[perm5[eval(sample["task_id"])[0]]].append(sample)


    ## create a dataloader for batch each of this
    test_dataset_by_predicted_id = {k: make_loader(args,v,model.tokenizer) for k,v in test_by_task_id.items()}

    for gold_task_id, task_loader in tqdm(test_dataset_by_predicted_id.items(),total=len(test_dataset_by_predicted_id)):
        print(f"Task Id: {gold_task_id}")
        for idx_b, batch in tqdm(enumerate(task_loader),total=len(task_loader)):
            with torch.no_grad():
                value_batch,_ = test_generation_GPT2BATCH(model=model.model,
                                                    tokenizer=model.tokenizer,
                                                    input_text=[b+"[SOS]" for b in batch['history']],
                                                    device=device,
                                                    max_length=100,
                                                    task_id=gold_task_id)
            for idx, resp in enumerate(value_batch):
                results.append({"id":batch["dial_id"][idx],"turn_id":batch["turn_id"][idx],
                                "dataset":batch["dataset"][idx],"task_id":batch["task_id"][idx],
                                "spk":batch["spk"][idx],"gold":batch["reply"][idx],
                                "genr":resp,"hist":batch["history"][idx],"pred_task_id":gold_task_id})
            
        # if(idx_b==1): break

    with open(f'{direct}/gold_id_response_perm_{permu}.json', 'w') as fp:
        json.dump(results, fp, indent=4)
