
import os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import quadprog
from random import sample
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel,T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from utils.data_seq2seq import get_data_loaders, get_current_task_data, make_loader
from utils.data import add_special_tokens_
from utils.parser import get_args
from test import test_model_seq2seq, generate_sample_prev_task, test_model_seq2seq_ADAPTER
from collections import defaultdict
from adapterGPT2 import GPT2Adapter
import torch.autograd as autograd
from prettytable import PrettyTable


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

class Seq2SeqToD(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        if "t5" in args.model_checkpoint:
            model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "bart" in args.model_checkpoint:
            model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "gpt2" in args.model_checkpoint:
            if(args.CL == "ADAPTER" or args.CL == "LOGADAPTER"):
                model = GPT2Adapter.from_pretrained(args.model_checkpoint)
                model.add_adapters(bottleneck_size=args.bottleneck_size,adapter_num=args.number_of_adpt,superposition=args.superposition)
            else:
                model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sos_token="[SOS]", sep_token="[sep]",pad_token='[PAD]')
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))

        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.lr
        self.current_task = 0
        self.fisher = defaultdict(list)
        self.optpar = defaultdict(list)
        self.episodic_mem = defaultdict(list)
        self.CL = args.CL
        self.superposition = args.superposition
        self.reg = args.reg
        self.first_task = True
        self.model_name = args.model_checkpoint
        self.reply_memory = []
        self.task_list_seen = []

    def set_number_of_tasks(self,n_tasks):
        self.n_tasks = n_tasks

    def set_up_gem(self):
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        dev = next(self.model.parameters()).device
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks).to(dev)

    def compute_task_SupSup(self,batch,device='cuda'):
        n_task = len(self.task_list_seen)
        task_ids_batch = []
        for b in batch["history"]:
            batched_history = self.tokenizer([b], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False,return_attention_mask=False)
            input_ids = batched_history['input_ids']
            batched_history_out = self.tokenizer([b], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False, return_attention_mask=False)
            batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids']==self.tokenizer.pad_token_id, -100)
            output_ids = batched_history_out['input_ids'] ### basically just remove pad from ppl calculation
            self.model.zero_grad()

            alphas = torch.ones(n_task) / n_task
            alphas.requires_grad_(True)

            (loss), *_ = self.model(
                    input_ids=input_ids.to(device),
                    attention_mask=None,
                    labels=output_ids.to(device),
                    task_id=-2,
                    alphas=alphas.to(device),
                    current_task_id=len(self.task_list_seen)
                    )
            # Gradient wrt alphas
            g, = autograd.grad(loss, alphas)
            inferred_task = (-g).squeeze().argmax()
            task_ids_batch.append(inferred_task.item())
        return task_ids_batch


    def compute_PPL(self,batch,task_id=-1,device='cuda'):
        with torch.no_grad():
            lm_logits, *_ = self.model(
                            input_ids=batch["input_id_PPL"].to(device),
                            attention_mask=None,
                            labels=None,
                            task_id=task_id
                            )
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch["output_id_PPL"].to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1)/(loss!=0).sum(1)).tolist()

    def training_step(self, batch, batch_idx):

        if self.CL == "GEM" and not self.first_task:
            dev = next(self.model.parameters()).device
            for id_task, (_,task_memory) in enumerate(self.episodic_mem.items()):
                batch_mem =  sample(task_memory,1)[0] # ==> we sample one batch from episodic memory
                self.model.zero_grad()
                (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                    attention_mask=batch_mem["attention_mask"].to(dev) if "gpt2" not in self.model_name else None,
                    labels=batch_mem["decoder_output"].to(dev)
                    )
                loss.backward()
                store_grad(self.model.parameters, self.grads, self.grad_dims, id_task)
            self.model.zero_grad()

        elif(self.CL == "AGEM" and not self.first_task):
            dev = next(self.model.parameters()).device
            batch_mem = sample(self.episodic_mem["all"],1)[0] # ==> we sample one batch from episodic memory
            self.model.zero_grad()
            (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                attention_mask=batch_mem["attention_mask"].to(dev) if "gpt2" not in self.model_name else None,
                labels=batch_mem["decoder_output"].to(dev)
                )
            loss.backward()
            grad_ref = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_ref.append(p.grad.view(-1))
            grad_ref = torch.cat(grad_ref) ## from eq. 10 of AGEM Paper

            self.model.zero_grad()


        ## LOSS ON CURRENT DATA
        if(self.CL == "ADAPTER"):
            (loss), *_ = self.model(
                                input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["decoder_output"],
                                task_id=self.task_list_seen.index(batch["task_id"][0])
                                )
        elif(self.CL == "LOGADAPTER"):
            (loss), *_ = self.model(
                                input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["decoder_output"],
                                task_id=int(batch["task_id"][0])
                                )
        else:
            (loss), *_ = self.model(input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"])

        if(self.CL == "AGEM" and not self.first_task):
            ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
            loss.backward()
            grad_cur = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur*grad_ref).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_ref*grad_ref).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_ref
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param
        elif self.CL == "GEM" and not self.first_task:
            loss.backward()
            store_grad(self.model.parameters, self.grads, self.grad_dims, id_task+1)
            indx = torch.LongTensor([j for j in range(id_task+1)])
            dotp = torch.mm(self.grads.to(dev)[:, id_task].unsqueeze(0), self.grads.to(dev).index_select(1, indx.to(dev)))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads.to(dev)[:, id_task].unsqueeze(1), self.grads.to(dev).index_select(1, indx.to(dev)), self.reg)
                # copy gradients back
                overwrite_grad(self.model.parameters, self.grads.to(dev)[:, id_task], self.grad_dims)

        elif self.CL == "L2" and not self.first_task:
            dev = next(self.model.parameters()).device
            l2_reg = 0

            for n,p in self.model.named_parameters():
                l = self.reg * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            self.log('l2_reg', l2_reg, on_epoch=True)
            loss = loss + l2_reg
        elif self.CL == "EWC" and not self.first_task:
            dev = next(self.model.parameters()).device
            ewc_loss = 0
            for n,p in self.model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                l = self.reg * self.fisher[n].to(dev) * (p - self.optpar[n].to(dev)).pow(2)
                ewc_loss += l.sum()
            self.log('EWC_reg', ewc_loss, on_epoch=True)
            loss = loss + ewc_loss

        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if(self.CL == "ADAPTER"):
            (loss), *_ = self.model(input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["decoder_output"],
                                task_id=self.task_list_seen.index(batch["task_id"][0])
                                )
        elif(self.CL == "LOGADAPTER"):
            (loss), *_ = self.model(
                                input_ids=batch["encoder_input"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["decoder_output"],
                                task_id=int(batch["task_id"][0])
                                )
        else:
            (loss), *_ = self.model(input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"]
                    )
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        if(self.CL=="ADAPTER"):
            if(self.superposition):
                parameters_to_update = [p for n, p in self.named_parameters() if "adapter" in str(n)]
                # parameters_to_update = [p for n, p in self.named_parameters() if "adapter" in str(n) and ".o" not in str(n)]
                # print([n for n, p in self.named_parameters() if "adapter" in str(n) and ".o" not in str(n)])
            else:
                parameters_to_update = [p for n, p in self.named_parameters() if "adapter" in str(n)]

            return AdamW(parameters_to_update, lr=self.lr, correct_bias=True)
        else:
            return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def backward(self, loss, optimizer, optimizer_idx):
        if (self.CL == "GEM" or self.CL == "AGEM") and not self.first_task:
            pass
        else:
            loss.backward()

def train():
    args = get_args()
    # train!

    model = Seq2SeqToD(args)
    train_loader, val_loader, dev_val_loader, (train_datasets, test_datasets) = get_data_loaders(args, model.tokenizer)

    task_seen_so_far = []
    if(args.CL != ""): model.set_number_of_tasks(len(list(train_loader.keys())))
    if(args.CL == "GEM"): model.set_up_gem()

    if args.multi:
        trainer = Trainer(
                default_root_dir=args.saving_dir,
                accumulate_grad_batches=args.gradient_accumulation_steps,
                gradient_clip_val=args.max_norm,
                max_epochs=args.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                gpus=list(map(int, args.GPU)),
                # limit_train_batches=10,
                # precision=16,
                # distributed_backend="dp"
                )
        trainer.fit(model, train_loader, val_loader)

        model.model.save_pretrained(f'{args.saving_dir}')
        model.tokenizer.save_pretrained(f'{args.saving_dir}')
        test_model_seq2seq(args,model.model,model.tokenizer,dev_val_loader,time=f"multi")
    elif args.continual:
        # test_model_seq2seq(args,model.model,model.tokenizer,dev_val_loader,time="0_['']")

        for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            model.task_list_seen.append(task_id)
            if (args.CL == "REPLAY"):
                print(f"Memory Size {len(model.reply_memory)}")
                task_loader = make_loader(args,train_datasets[task_id]+model.reply_memory,model.tokenizer)
            if(args.CL == "LAML"):
                number_of_sample = int(len(train_datasets[task_id])*args.percentage_LAML)
                aug_current_task = get_current_task_data(args,train_datasets,task_id,number_of_sample)
                print(f"Current {task_id} AUG: {len(aug_current_task)}")
                aug_data_prev_task = []
                for task_id_so_far in task_seen_so_far:
                    ## sample data by the LM, priming with [task_id] e.g., [hotel]
                    temp = generate_sample_prev_task(args,model.model,model.tokenizer,train_datasets,task_id_so_far,number_of_sample,time=f"{task_num}_{task_id}")
                    print(f"Current {task_id_so_far} AUG: {len(temp)}")
                    aug_data_prev_task += temp
                ## this task_loader include data generated by the same model
                task_loader = make_loader(args,train_datasets[task_id]+aug_current_task+aug_data_prev_task,model.tokenizer)

            trainer = Trainer(
                default_root_dir=f'{args.saving_dir}/{task_num}_{task_id}',
                accumulate_grad_batches=args.gradient_accumulation_steps,
                gradient_clip_val=args.max_norm,
                max_epochs=args.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=False, mode='min')],
                gpus=list(map(int, args.GPU)),
                # logger=comet_logger,
                # limit_train_batches=10,
                # checkpoint_callback=False
            )
            trainer.fit(model, task_loader, val_loader[task_id])
            #load best model
            if(args.CL != "LAML" and args.CL != "ADAPTER"):
                checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
                print("load from:",trainer.checkpoint_callback.best_model_path)
                checkpoint['state_dict'] = { k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() }
                model.model.load_state_dict(checkpoint['state_dict'])

            if(args.CL == "ADAPTER"):
                test_model_seq2seq_ADAPTER(args,model,model.tokenizer,dev_val_loader,test_datasets,time=f"{task_num}_{task_id}")
            else:   
                test_model_seq2seq(args,model.model,model.tokenizer,dev_val_loader,time=f"{task_num}_{task_id}")

            model.first_task = False
            ## save some training data into the episodic mem
            if args.CL == "AGEM":
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem["all"].append(b)
                    if idx_b==args.episodic_mem_size: break
            elif args.CL == "REPLAY":
                model.reply_memory += sample(train_datasets[task_id],min(len(train_datasets[task_id]),args.episodic_mem_size*args.train_batch_size))
            else: ## save example per task
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem[task_id].append(b)
                    if idx_b==args.episodic_mem_size: break


            ##### Compute Fisher info Matrix for EWC
            if args.CL == "EWC" or args.CL =="L2":
                model.model.cpu()
                for n, p in model.model.named_parameters():
                    model.optpar[n] = torch.Tensor(p.cpu().data)
                    model.fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()

                if args.CL == "EWC":
                    print("COMPUTING Fisher Information Matrix")
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
                        model.fisher[name_f] /= len(model.episodic_mem[task_id]) #*args.train_batch_size
                    model.model.zero_grad()
                    print("DONE COMPUTING Fisher Information Matrix")

            task_seen_so_far.append(task_id)
        model.model.save_pretrained(f'{args.saving_dir}')
        model.tokenizer.save_pretrained(f'{args.saving_dir}')


if __name__ == "__main__":
    train()
