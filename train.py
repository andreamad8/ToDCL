import torch
import json
import os
import os.path
import math
import glob
import re

import time
from random import sample
import pytorch_lightning as pl
import random
from pytorch_lightning import Trainer, seed_everything
from utils.dataloader import (
    get_data_loaders,
    get_current_task_data,
    make_loader,
    ValEveryNSteps,
)

from test import (
    test_model_seq2seq,
    generate_sample_prev_task,
    test_model_seq2seq_ADAPTER,
)
from collections import defaultdict
from CL_learner import Seq2SeqToD

from argparse import ArgumentParser

import subprocess
import numpy as np

import warnings
import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def get_free_gpu(num_gpu):
    cmd = "nvidia-smi -q -d pids |grep -A4 GPU|grep Processes >tmp"
    p = subprocess.Popen(["/bin/bash", "-c", cmd])
    p.wait()
    memory_available = [x.split(":")[-1].strip() for x in open("tmp", "r").readlines()]
    ids = []
    for index in range(len(memory_available)):
        if memory_available[index] == "None":
            ids.append(str(index))
    print("Allocating Model to " + str(ids[:num_gpu]))
    return ",".join(ids[:num_gpu])


os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu(1)
reserve = torch.tensor(1)
if torch.cuda.is_available():
    reserve.to("cuda:0")


def get_checkpoint(log_dir, index_to_load):
    file = glob.glob(f"{log_dir}/*")
    for f in file:
        f_noprefix = f.replace(f"{log_dir}", "")
        num = [int(s) for s in re.findall(r"\d+", f_noprefix)]
        if index_to_load in num:
            version = os.listdir(f + "/lightning_logs")[0]
            check_name = os.listdir(f + "/lightning_logs/" + version + "/checkpoints/")[
                0
            ]
            # checkpoint_name = f.replace("[","\[").replace("]","\]").replace("\'","\\'")+"/lightning_logs/"+ version+"/checkpoints/"+check_name
            checkpoint_name = (
                f + "/lightning_logs/" + version + "/checkpoints/" + check_name
            )
    return checkpoint_name


def train(hparams, *args):
    if hparams.CL == "ADAPTER":
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    else:
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}/{hparams.CL}_EM_{hparams.episodic_mem_size}_LAMOL_{hparams.percentage_LAM0L}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    if hparams.CL == "MULTI":
        hparams.multi = True
        hparams.continual = False
    else:
        hparams.multi = False
        hparams.continual = True

    # train!
    model = Seq2SeqToD(hparams)

    (
        train_loader,
        val_loader,
        dev_val_loader,
        (train_datasets, val_datasets, test_datasets),
    ) = get_data_loaders(hparams, model.tokenizer)

    ## make the permutation
    if hparams.continual:
        seed_everything(hparams.seed)
        keys = list(train_loader.keys())
        random.shuffle(keys)
        train_loader = {key: train_loader[key] for key in keys}
        print(f"RUNNING WITH SEED {hparams.seed}")
        for k, _ in train_loader.items():
            print(k)
        print()

    task_seen_so_far = []
    if hparams.CL != "MULTI":
        model.set_number_of_tasks(len(list(train_loader.keys())))
    if hparams.CL == "GEM":
        model.set_up_gem()


    if hparams.multi:
        start = time.time()
        train_parameters = {
            'default_root_dir': hparams.saving_dir,
            'accumulate_grad_batches': hparams.gradient_accumulation_steps,
            'gradient_clip_val': hparams.max_norm,
            'max_epochs': hparams.n_epochs,
            'callbacks': [
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=5,
                    verbose=False,
                    mode="min",
                ),
            ],
        }
        if torch.cuda.is_available():
            train_parameters['gpus'] = [0]




        trainer = Trainer(**train_parameters)
        trainer.fit(model, train_loader, val_loader)
        end = time.time()
        print("Time elapsed:", end - start)
        model.model.save_pretrained(f"{hparams.saving_dir}")
        model.tokenizer.save_pretrained(f"{hparams.saving_dir}")
        test_model_seq2seq(
            hparams, model.model, model.tokenizer, dev_val_loader, time=f"FINAL"
        )
    elif hparams.continual:
        for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            model.task_list_seen.append(task_id)

            if hparams.CL == "REPLAY":
                print(f"Memory Size {len(model.reply_memory)}")
                task_loader = make_loader(
                    hparams,
                    train_datasets[task_id] + model.reply_memory,
                    model.tokenizer,
                )

            if hparams.CL == "LAMOL":
                if current_task_to_load == None or task_num >= current_task_to_load:
                    number_of_sample = hparams.percentage_LAM0L
                    aug_current_task = get_current_task_data(
                        hparams, train_datasets[task_id], task_id, number_of_sample
                    )
                    print(f"Current {task_id} AUG: {len(aug_current_task)}")
                    aug_data_prev_task = []
                    for task_id_so_far in task_seen_so_far:
                        ## sample data by the LM, priming with [task_id] e.g., [hotel]
                        temp = generate_sample_prev_task(
                            hparams,
                            model.model,
                            model.tokenizer,
                            train_datasets,
                            task_id_so_far,
                            number_of_sample,
                            time=f"{task_num}_{task_id}",
                        )
                        print(f"Current {task_id_so_far} AUG: {len(temp)}")
                        aug_data_prev_task += temp
                    ## this task_loader include data generated by the same model
                    task_loader = make_loader(
                        hparams,
                        train_datasets[task_id] + aug_current_task + aug_data_prev_task,
                        model.tokenizer,
                    )

            ## CORE
            print()
            print(f"TASK:{task_id}")
            start = time.time()
            train_parameters = {
                'default_root_dir': f"{hparams.saving_dir}/{task_num}_{task_id}",
                'accumulate_grad_batches': hparams.gradient_accumulation_steps,
                'gradient_clip_val': hparams.max_norm,
                'max_steps': hparams.n_steps,
                'max_epochs': 1000,
                'check_val_every_n_epoch': 1000,
                'callbacks': [
                    ValEveryNSteps(20),
                    pl.callbacks.ModelCheckpoint(
                        monitor="val_loss", save_on_train_epoch_end=False
                    ),
                    pl.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=0.00,
                        patience=5,
                        verbose=True,
                        mode="min",
                        check_on_train_epoch_end=False,
                    ),
                ],
            }
            if torch.cuda.is_available():
                train_params['gpus'] = [0]

            trainer = Trainer(
                **train_parameters
            )
            trainer.fit(model, task_loader, val_loader[task_id])
            end = time.time()
            print("Time elapsed:", end - start)
            # load best model
            # this model are better if the are runned to they epoch number
            if hparams.CL != "LAMOL" and hparams.CL != "EWC":
                # checkpoint = torch.load(trainer.checkpoint_callback.best_model_path) use this if the next doesn't work
                checkpoint = torch.load(
                    trainer.checkpoint_callback.best_model_path,
                    map_location=lambda storage, loc: storage,
                )
                print("load from:", trainer.checkpoint_callback.best_model_path)
                checkpoint["state_dict"] = {
                    k.replace("model.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
                model.model.load_state_dict(checkpoint["state_dict"])

            # testing the model by generating the answers
            if hparams.test_every_step:
                if hparams.CL == "ADAPTER":
                    test_model_seq2seq_ADAPTER(
                        hparams,
                        model,
                        model.tokenizer,
                        dev_val_loader,
                        test_datasets,
                        time=f"{task_num}_{task_id}",
                    )
                else:
                    test_model_seq2seq(
                        hparams,
                        model.model,
                        model.tokenizer,
                        dev_val_loader,
                        time=f"{task_num}_{task_id}",
                    )

            ## END CORE

            model.first_task = False
            ## save some training data into the episodic mem
            if hparams.CL == "AGEM":
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem["all"].append(b)
                    if idx_b == hparams.episodic_mem_size:
                        break
            elif hparams.CL == "REPLAY":
                # in percentage
                model.reply_memory += sample(
                    train_datasets[task_id],
                    min(len(train_datasets[task_id]), hparams.episodic_mem_size),
                )  # sample(train_datasets[task_id],min(len(train_datasets[task_id]),int(hparams.episodic_mem_size*len(train_datasets[task_id])))
            else:  ## save example per task
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem[task_id].append(b)
                    if idx_b == hparams.episodic_mem_size:
                        break

            ##### Compute Fisher info Matrix for EWC
            if hparams.CL == "EWC" or hparams.CL == "L2":
                model.model.cpu()
                for n, p in model.model.named_parameters():
                    model.optpar[n] = torch.Tensor(p.cpu().data)
                    model.fisher[n] = torch.zeros(
                        p.size()
                    )  # torch.Tensor(p.cpu().data).zero_()

                if hparams.CL == "EWC":
                    for _, batch in enumerate(model.episodic_mem[task_id]):
                        model.model.zero_grad()
                        (loss), *_ = model.model(
                            input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"],
                        )
                        loss.backward()
                        for n, p in model.model.named_parameters():
                            if p.grad is not None:
                                model.fisher[n].data += p.grad.data**2

                    for name_f, _ in model.fisher.items():
                        model.fisher[name_f] /= len(
                            model.episodic_mem[task_id]
                        )  # *hparams.train_batch_size
                    model.model.zero_grad()
            task_seen_so_far.append(task_id)

        model.model.save_pretrained(f"{hparams.saving_dir}")
        model.tokenizer.save_pretrained(f"{hparams.saving_dir}")
        if hparams.CL == "ADAPTER":
            test_model_seq2seq_ADAPTER(
                hparams,
                model,
                model.tokenizer,
                dev_val_loader,
                test_datasets,
                time=f"FINAL",
            )
        else:
            test_model_seq2seq(
                hparams, model.model, model.tokenizer, dev_val_loader, time=f"FINAL"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="t5-base")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=10, help="Batch size for validation"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size for validation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument(
        "--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving"
    )
    parser.add_argument(
        "--max_history", type=int, default=5, help="max number of turns in the dialogue"
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action="store_true", help="continual baseline")
    parser.add_argument(
        "--test_every_step", action="store_true", help="continual baseline"
    )
    parser.add_argument(
        "--length", type=int, default=50, help="lenght of the generation"
    )
    parser.add_argument("--debug", action="store_true", help="continual baseline")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of training steps in continual",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="Number of training steps in multi"
    )

    parser.add_argument("--bottleneck_size", type=int, default=100)
    parser.add_argument(
        "--number_of_adpt", type=int, default=40, help="number of adapterss"
    )
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument(
        "--percentage_LAM0L",
        type=float,
        default=0.2,
        help="LAMOL percentage of augmented data used",
    )
    parser.add_argument(
        "--reg", type=float, default=0.01, help="CL regularization term"
    )
    parser.add_argument(
        "--episodic_mem_size",
        type=int,
        default=100,
        help="number of batch/sample put in the episodic memory",
    )
    #  options=["E2E","DST","NLG","INTENT"]
    parser.add_argument("--task_type", type=str, default="NLG")
    #  options=["VANILLA"]
    parser.add_argument("--CL", type=str, default="MULTI")
    # options=[1,2,3,4,5]
    parser.add_argument("--seed", default=1, type=int)

    hyperparams = parser.parse_args()
    train(hyperparams)
