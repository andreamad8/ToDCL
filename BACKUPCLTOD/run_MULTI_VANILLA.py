"""Example launcher for a hyperparameter search on SLURM.
This example shows how to use gpus on SLURM with PyTorch.
"""
import os
import torch
import numpy as np
import quadprog
import json
import os.path
import math
import glob
import re
import os

import time
from random import sample
import pytorch_lightning as pl
import random
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel,T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from utils.data_seq2seq import get_data_loaders, get_current_task_data, make_loader, update_task_id
from utils.data import add_special_tokens_
from utils.parser import get_args
from test import test_model_seq2seq, generate_sample_prev_task, test_model_seq2seq_ADAPTER
from collections import defaultdict

from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
from train_CL import store_grad, overwrite_grad, project2cone2, Seq2SeqToD


def get_checkpoint(log_dir, index_to_load):
    file = glob.glob(f"{log_dir}/*")
    for f in file:
        f_noprefix = f.replace(f"{log_dir}","")
        num = [int(s) for s in re.findall(r'\d+', f_noprefix)]
        if index_to_load in num:
            version = os.listdir(f+"/lightning_logs")[0]
            check_name = os.listdir(f+"/lightning_logs/"+ version+"/checkpoints/")[0]
            # checkpoint_name = f.replace("[","\[").replace("]","\]").replace("\'","\\'")+"/lightning_logs/"+ version+"/checkpoints/"+check_name
            checkpoint_name = f+"/lightning_logs/"+ version+"/checkpoints/"+check_name
    return checkpoint_name

def get_max_taskid(task_number):
    return math.floor(math.log(task_number,2))


def train(hparams, *args):
    if(hparams.CL == "LOGADAPTER"):
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_EM_{hparams.episodic_mem_size}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_COMPRESSING_{hparams.compressing_factor}_{hparams.model_checkpoint}"
    elif(hparams.CL == "ADAPTER"):
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    else:
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}/{hparams.CL}_EM_{hparams.episodic_mem_size}_LAML_{hparams.percentage_LAML}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    if(hparams.CL == "MULTI"): 
        hparams.multi = True
        hparams.continual = False
    else: 
        hparams.multi = False
        hparams.continual = True

    # train!
    model = Seq2SeqToD(hparams)

    if(hparams.CL == "LOGADAPTER"):
        model.reply_memory = defaultdict(list)

    ### check if the task has to be loaded 
    current_task_to_load = None
    if(os.path.isfile(f'{hparams.saving_dir}'+'/current_task.json')):
        current_tasks = json.load(open(f'{hparams.saving_dir}'+'/current_task.json'))
        current_task_to_load = int(current_tasks[0])
        # print(current_task_to_load)
        if current_task_to_load!= 0:
            load_check_point = get_checkpoint(hparams.saving_dir,current_task_to_load-1)
            # files = glob.glob(f"{hparams.saving_dir}/*")
            checkpoint = torch.load(load_check_point)
            print("load from:",load_check_point)
            checkpoint['state_dict'] = { k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() }
            model.model.load_state_dict(checkpoint['state_dict'])

    if((hparams.CL == "ADAPTER" or hparams.CL == "REPLAY" ) and hparams.load_check_point!= "None"):
        checkpoint = torch.load(hparams.load_check_point)
        print("load from:",hparams.load_check_point)
        checkpoint['state_dict'] = { k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() }
        model.model.load_state_dict(checkpoint['state_dict'])

    train_loader, val_loader, dev_val_loader, (train_datasets, val_datasets, test_datasets) = get_data_loaders(hparams, model.tokenizer)

    ## make the permutation
    if(hparams.continual):
        seed_everything(hparams.seed)
        keys =  list(train_loader.keys())
        random.shuffle(keys)
        train_loader = {key: train_loader[key] for key in keys}
        print(f"RUNNING WITH SEED {hparams.seed}")
        for k,_ in train_loader.items():
            print(k)
        print()


    task_seen_so_far = []
    map_adpt_to_task = defaultdict(list)
    current_adpt_num = -1
    if(hparams.CL != "MULTI"): model.set_number_of_tasks(len(list(train_loader.keys())))
    if(hparams.CL == "GEM"): model.set_up_gem()

    if hparams.multi:
        start = time.time()
        trainer = Trainer(
                default_root_dir=hparams.saving_dir,
                accumulate_grad_batches=hparams.gradient_accumulation_steps,
                gradient_clip_val=hparams.max_norm,
                max_epochs=hparams.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                gpus=[0],
                # gpus=[0,1,2,3,4,5,6,7],
                # limit_train_batches=10,
                precision=16,
                # distributed_backend="dp",
                # progress_bar_refresh_rate=0.1,
                )
        trainer.fit(model, train_loader, val_loader)
        end = time.time()
        print ("Time elapsed:", end - start)
        model.model.save_pretrained(f'{hparams.saving_dir}')
        model.tokenizer.save_pretrained(f'{hparams.saving_dir}')
        test_model_seq2seq(hparams,model.model,model.tokenizer,dev_val_loader,time=f"multi")
    elif hparams.continual:
    #     # test_model_seq2seq(hparams,model.model,model.tokenizer,dev_val_loader,time="0_['']")

        for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            model.task_list_seen.append(task_id)

            if not os.path.exists(f'{hparams.saving_dir}'):
                os.makedirs(f'{hparams.saving_dir}')
            with open(f'{hparams.saving_dir}'+'/current_task.json', 'w') as fp:
                json.dump([task_num], fp, indent=4)

            if((hparams.CL == "ADAPTER" or hparams.CL == "REPLAY" ) and hparams.load_check_point!= "None"):
                if task_num <= hparams.current_task_to_load:
                    model.reply_memory += train_datasets[task_id]
                    continue
            if(hparams.CL == "REPLAY"):
                print(f"Memory Size {len(model.reply_memory)}")
                task_loader = make_loader(hparams,train_datasets[task_id]+model.reply_memory,model.tokenizer)

            if(hparams.CL == "LOGADAPTER"):
                print("CURRENT TASK",task_num)
                new_task_id = math.floor(task_num/hparams.compressing_factor)
                current_adpt_num = new_task_id
                model.reply_memory[new_task_id] += sample(train_datasets[task_id],min(len(train_datasets[task_id]),hparams.episodic_mem_size))
                print(f"MEMORY TASK {new_task_id} SIZE:",len(model.reply_memory[new_task_id]))
                map_adpt_to_task[new_task_id].append(task_id)
                print("MAP:",map_adpt_to_task)
                print()
                print()
                # aug_data_prev_task = []
                # number_of_sample = 100
                # max_adpt = get_max_taskid(task_num+1)## task_num+1 because our loop goes from 0.... and the other is in the eq
                # print("MAX ADPT",max_adpt)
                # if max_adpt > current_adpt_num:
                #     print("SPAWN A NEW ADAPTER")
                #     new_task_id = max_adpt
                #     current_adpt_num += 1
                #     print("TOTAL ADAPTER",current_adpt_num)
                # else: 
                #     print("USE EXISTING ONE")
                #     ## for now we choose it random
                #     new_task_id = random.randint(1,max_adpt)#get_best_adpater() ## THIS IS THE ID OF THE ADAPTER WE HAVE IN MEMORY THAT MATCH THE MOST OUR DATA
                #     print("NEW TASK ID",new_task_id)
                #     for task_id_seen_by_adpt in map_adpt_to_task[new_task_id]:
                #         temp = generate_sample_prev_task(hparams,model.model,model.tokenizer,train_datasets,task_id_seen_by_adpt,number_of_sample,time=f"{task_num}_{task_id}",task_id_adpt=new_task_id)
                #         print(f"Current {task_id_seen_by_adpt} AUG: {len(temp)}")
                #         temp = update_task_id(hparams,temp,new_task_id)
                #         aug_data_prev_task += temp

                ## UPDATE TASK_ID ADPATER 
           
                data_train = update_task_id(hparams,train_datasets[task_id],new_task_id)
                val_loader[task_id] = make_loader(hparams,update_task_id(hparams,val_datasets[task_id],new_task_id),model.tokenizer)
                # aug_current_task = get_current_task_data(hparams,data_train,task_id,number_of_sample)
                task_loader = make_loader(hparams,data_train+model.reply_memory[new_task_id],model.tokenizer)


            if(hparams.CL == "LAML"):
                if(current_task_to_load == None or task_num >= current_task_to_load):
                    number_of_sample = hparams.percentage_LAML #int(len(train_datasets[task_id])*hparams.percentage_LAML)
                    aug_current_task = get_current_task_data(hparams,train_datasets[task_id],task_id,number_of_sample)
                    print(f"Current {task_id} AUG: {len(aug_current_task)}")
                    aug_data_prev_task = []
                    for task_id_so_far in task_seen_so_far:
                        ## sample data by the LM, priming with [task_id] e.g., [hotel]
                        temp = generate_sample_prev_task(hparams,model.model,model.tokenizer,train_datasets,task_id_so_far,number_of_sample,time=f"{task_num}_{task_id}")
                        print(f"Current {task_id_so_far} AUG: {len(temp)}")
                        aug_data_prev_task += temp
                    ## this task_loader include data generated by the same model
                    task_loader = make_loader(hparams,train_datasets[task_id]+aug_current_task+aug_data_prev_task,model.tokenizer)


            ## this is for loading purposes
            if(current_task_to_load == None or task_num >= current_task_to_load):
                ## CORE
                trainer = Trainer(
                    default_root_dir=f'{hparams.saving_dir}/{task_num}_{task_id}',
                    accumulate_grad_batches=hparams.gradient_accumulation_steps,
                    gradient_clip_val=hparams.max_norm,
                    max_epochs=hparams.n_epochs,
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='min')],
                    gpus=[0],
                    # gpus=[0,1,2,3],
                    # limit_train_batches=100,
                    # precision=16,
                    # distributed_backend="dp",
                    # progress_bar_refresh_rate=0.1
                )
                trainer.fit(model, task_loader, val_loader[task_id])
                #load best model
                # this model are better if the are runned to they epoch number
                # and hparams.CL != "ADAPTER",and hparams.CL != "LOGADAPTER"
                if(hparams.CL != "LAML" and hparams.CL != "LOGADAPTER" and hparams.CL != "EWC"):
                    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
                    print("load from:",trainer.checkpoint_callback.best_model_path)
                    checkpoint['state_dict'] = { k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() }
                    model.model.load_state_dict(checkpoint['state_dict'])

                # testing the model by generating the answers
                if(hparams.CL == "ADAPTER" or hparams.CL == "LOGADAPTER"):
                    test_model_seq2seq_ADAPTER(hparams,model,model.tokenizer,dev_val_loader,test_datasets,time=f"{task_num}_{task_id}",max_seen_task=current_adpt_num)
                else:                
                    test_model_seq2seq(hparams,model.model,model.tokenizer,dev_val_loader,time=f"{task_num}_{task_id}")

                ## END CORE
            else:
                print(f"Skipping task: {task_num}")

            model.first_task = False
            ## save some training data into the episodic mem
            if hparams.CL == "AGEM":
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem["all"].append(b)
                    if idx_b==hparams.episodic_mem_size: break
            elif hparams.CL == "REPLAY":
                # in percentage
                model.reply_memory += sample(train_datasets[task_id],min(len(train_datasets[task_id]),hparams.episodic_mem_size))# sample(train_datasets[task_id],min(len(train_datasets[task_id]),int(hparams.episodic_mem_size*len(train_datasets[task_id])))
                # model.reply_memory += train_datasets[task_id]#sample(train_datasets[task_id],min(len(train_datasets[task_id]),hparams.episodic_mem_size*hparams.train_batch_size))
            else: ## save example per task
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem[task_id].append(b)
                    if idx_b==hparams.episodic_mem_size: break


            ##### Compute Fisher info Matrix for EWC
            if hparams.CL == "EWC" or hparams.CL =="L2":
                model.model.cpu()
                for n, p in model.model.named_parameters():
                    model.optpar[n] = torch.Tensor(p.cpu().data)
                    model.fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()

                if hparams.CL == "EWC":
                    for _, batch in enumerate(model.episodic_mem[task_id]):
                        model.model.zero_grad()
                        (loss), *_ = model.model(input_ids=batch["encoder_input"],
                                                attention_mask=batch["attention_mask"],
                                                labels=batch["decoder_output"])
                        loss.backward()
                        for n, p in model.model.named_parameters():
                            if p.grad is not None:
                                model.fisher[n].data += p.grad.data ** 2

                    for name_f,_ in model.fisher.items():
                        model.fisher[name_f] /= len(model.episodic_mem[task_id]) #*hparams.train_batch_size
                    model.model.zero_grad()
            task_seen_so_far.append(task_id)



        model.model.save_pretrained(f'{hparams.saving_dir}')
        model.tokenizer.save_pretrained(f'{hparams.saving_dir}')



if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving")
    parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    # parser.add_argument("--CL", type=str, default="", help="CL strategy used")
    parser.add_argument("--percentage_LAML", type=float, default=0.2, help="LAML percentage of augmented data used")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--episodic_mem_size", type=int, default=20, help="number of batch put in the episodic memory")
    parser.add_argument("--superposition", action='store_true', help="multitask baseline")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    # parser.add_argument("--task_type", type=str, default="E2E", help="Select the kind of task to run")

    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log_path', default='.')

    parser.opt_list('--task_type', type=str, default="E2E", options=["DST"], tunable=True)
    # parser.opt_list('--task_type', type=str, default="E2E", options=["E2E","DST","NLG","INTENT"], tunable=True)
    parser.opt_list('--model_checkpoint', type=str, default="gpt2", options=["gpt2"], tunable=True)
    parser.opt_list('--CL', type=str, default="", options=["MULTI","VANILLA"], tunable=True)
    # parser.opt_list('--CL', type=str, default="", options=["VANILLA"], tunable=True)
    parser.opt_list('--seed', default=1, type=int, options=[1,2,3,4,5], tunable=True)
    # parser.opt_list('--seed', default=1, type=int, options=[2], tunable=True)

    hyperparams = parser.parse_args()

    # # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python'
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='andreamad8@fb.com', on_done=False, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules([
        'cuda/10.1',
        'cudnn/v7.6.5.32-cuda.10.1',
        'anaconda3'
    ])

    # # Add commands to the non-SLURM portion.
    cluster.add_command('source activate CLTOD')

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    cluster.add_slurm_cmd(cmd='cpus-per-task', value='10', comment='CPUS per task.')
    cluster.add_slurm_cmd(cmd='mem', value='0', comment='Memory')
    cluster.add_slurm_cmd(cmd='constraint', value='volta32gb', comment='GPU type per task.')
    cluster.add_slurm_cmd(cmd='time', value='72:0:0', comment='GPU type per task.')

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        train,
        nb_trials=10,
        # This is what will display in the slurm queue:
        job_name='MULTI_VANILLA')