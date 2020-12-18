#!/bin/bash

python run_MULTI_VANILLA.py
python run_REPLAY.py
python run_LAML.py
python run_EWCL2.py
python run_AGEM.py
python run_ADAPTER.py


## E2E experiments

# ### GPT2
# python train_CL.py --saving_dir runs/MULTIGPT_SGD --multi --n_epochs 5 --GPU 0 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32
# python train_CL.py --saving_dir runs/VANILLAGPT_SGD --continual --n_epochs 5 --GPU 1 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32
# python train_CL.py --saving_dir runs/REPLAYGPT_SGD --continual --n_epochs 5 --GPU 4 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL REPLAY
# python train_CL.py --saving_dir runs/EWCGPT_SGD --continual --n_epochs 5 --GPU 0 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL EWC --train_batch_size 2 --episodic_mem_size 80
# python train_CL.py --saving_dir runs/AGEMGPT_SGD --continual --n_epochs 5 --GPU 1 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL AGEM --train_batch_size 4

# ## T5
# python train_CL.py --saving_dir runs/MULTIT5_SGD --multi --n_epochs 5 --GPU 2 --dataset_list SGD --model_checkpoint t5-small --valid_batch_size 32 --lr 6.25e-4
# python train_CL.py --saving_dir runs/VANILLAT5_SGD --continual --n_epochs 5 --GPU 2 --dataset_list SGD --model_checkpoint t5-small --valid_batch_size 32 --lr 6.25e-4
# python train_CL.py --saving_dir runs/REPLAYT5_SGD --continual --n_epochs 5 --GPU 2 --dataset_list SGD --model_checkpoint t5-small --valid_batch_size 32 --lr 6.25e-4 --CL REPLAY
# python train_CL.py --saving_dir runs/EWCT5_SGD --continual --n_epochs 5 --GPU 1 --dataset_list SGD --model_checkpoint t5-small --valid_batch_size 32 --CL EWC --lr 6.25e-4
# python train_CL.py --saving_dir runs/AGEMT5_SGD --continual --n_epochs 5 --GPU 3 --dataset_list SGD --model_checkpoint t5-small --valid_batch_size 32 --CL AGEM --lr 6.25e-4

# ## GPT2 LAML
# python train_CL.py --saving_dir runs/LAMLGPT2_SGD --continual --n_epochs 5 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL LAML --GPU 3


# ## NLG
# python train_CL.py --saving_dir runs_NLG/MULTIGPT_SGD --multi --n_epochs 5 --GPU 1 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --task_type NLG
# python train_CL.py --saving_dir runs_NLG/VANILLAGPT_SGD --continual --n_epochs 5 --GPU 2 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --task_type NLG
# python train_CL.py --saving_dir runs_NLG/REPLAYGPT_SGD --continual --n_epochs 5 --GPU 4 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL REPLAY --task_type NLG
# python train_CL.py --saving_dir runs_NLG/EWCGPT_SGD --continual --n_epochs 5 --GPU 4 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL EWC --train_batch_size 2 --episodic_mem_size 80 --task_type NLG
# python train_CL.py --saving_dir runs_NLG/AGEMGPT_SGD --continual --n_epochs 5 --GPU 5 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL AGEM --train_batch_size 4 --episodic_mem_size 40 --task_type NLG


# python train_CL.py --saving_dir runs/EWCGPT_SGD --continual --n_epochs 5 --GPU 0 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL EWC --train_batch_size 2 --episodic_mem_size 80



# python train_CL.py --saving_dir runs/TEST --continual --n_epochs 1 --GPU 0 --dataset_list SGD --model_checkpoint gpt2 --valid_batch_size 32 --CL ADAPTER
