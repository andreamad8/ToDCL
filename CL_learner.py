import os
import torch
from torch.nn import CrossEntropyLoss
from random import sample
import pytorch_lightning as pl
import logging
logging.basicConfig()

from transformers import (
    AdamW,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    T5Tokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    HoulsbyConfig,
)
from utils.dataloader import get_data_loaders, get_current_task_data, make_loader
from collections import defaultdict

class Seq2SeqToD(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if "t5" in args.model_checkpoint:
            model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(
                args.model_checkpoint,
                bos_token="[bos]",
                eos_token="[eos]",
                sep_token="[sep]",
            )
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "bart" in args.model_checkpoint:
            model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = BartTokenizer.from_pretrained(
                args.model_checkpoint,
                bos_token="[bos]",
                eos_token="[eos]",
                sep_token="[sep]",
            )
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "gpt2" in args.model_checkpoint:
            model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
            tokenizer = GPT2Tokenizer.from_pretrained(
                args.model_checkpoint,
                bos_token="[bos]",
                eos_token="[eos]",
                sos_token="[SOS]",
                sep_token="[sep]",
                pad_token="[PAD]",
            )
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        if args.CL == "ADAPTER":
            reduction = round(model.config.d_model / args.bottleneck_size)
            adapter_config = HoulsbyConfig(reduction_factor=reduction)
            for i in range(args.number_of_adpt):
                model.add_adapter(str(i), config=adapter_config)

        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.lr
        self.current_task = 0
        self.fisher = defaultdict(list)
        self.optpar = defaultdict(list)
        self.episodic_mem = defaultdict(list)
        self.CL = args.CL
        self.reg = args.reg
        self.first_task = True
        self.model_name = args.model_checkpoint
        self.reply_memory = []
        self.task_list_seen = []

    def set_number_of_tasks(self, n_tasks):
        self.n_tasks = n_tasks

    def set_up_gem(self):
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        dev = next(self.model.parameters()).device
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks).to(dev)

    def compute_PPL(self, batch, task_id=-1, device="cuda", tokenizer=None):
        # To Implement

        logging.error(f"In Compute PPL with: batch ({batch}), task_id: ({task_id}), and tokenizer ({tokenizer})")
        with torch.no_grad():
            model_out = self.model(
                input_ids=batch["input_id_PPL"].to(device),
                attention_mask=None,
                labels=None,
                return_dict = True
            )
            logging.warn(f'Model Out is: {model_out}')

        shift_logits = model_out.logits[..., :-1, :].contiguous()
        shift_labels = batch["output_id_PPL"].to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1)/(loss!=0).sum(1)).tolist()

    def training_step(self, batch, batch_idx):
        if self.CL == "GEM" and not self.first_task:
            dev = next(self.model.parameters()).device
            for id_task, (_, task_memory) in enumerate(self.episodic_mem.items()):
                batch_mem = sample(task_memory, 1)[
                    0
                ]  # ==> we sample one batch from episodic memory
                self.model.zero_grad()
                loss = self.model(
                    input_ids=batch_mem["encoder_input"].to(dev),
                    attention_mask=batch_mem["attention_mask"].to(dev)
                    if "gpt2" not in self.model_name
                    else None,
                    labels=batch_mem["decoder_output"].to(dev),
                )[0]
                loss.backward()
                store_grad(self.model.parameters, self.grads, self.grad_dims, id_task)
            self.model.zero_grad()

        elif self.CL == "AGEM" and not self.first_task:
            dev = next(self.model.parameters()).device
            batch_mem = sample(self.episodic_mem["all"], 1)[
                0
            ]  # ==> we sample one batch from episodic memory
            self.model.zero_grad()
            loss = self.model(
                input_ids=batch_mem["encoder_input"].to(dev),
                attention_mask=batch_mem["attention_mask"].to(dev)
                if "gpt2" not in self.model_name
                else None,
                labels=batch_mem["decoder_output"].to(dev),
            )[0]
            loss.backward()
            grad_ref = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_ref.append(p.grad.view(-1))
            grad_ref = torch.cat(grad_ref)  ## from eq. 10 of AGEM Paper

            self.model.zero_grad()

        ## LOSS ON CURRENT DATA
        if self.CL == "ADAPTER":
            task_id = str(self.task_list_seen.index(batch["task_id"][0]))
            self.model.train_adapter(task_id)
            self.model.set_active_adapters(task_id)
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )[0]

        if self.CL == "AGEM" and not self.first_task:
            ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
            loss.backward()
            grad_cur = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_ref).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_ref * grad_ref).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_ref
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index : index + n_param].view_as(p))
                        index += n_param
        elif self.CL == "GEM" and not self.first_task:
            loss.backward()
            store_grad(self.model.parameters, self.grads, self.grad_dims, id_task + 1)
            indx = torch.LongTensor([j for j in range(id_task + 1)])
            dotp = torch.mm(
                self.grads.to(dev)[:, id_task].unsqueeze(0),
                self.grads.to(dev).index_select(1, indx.to(dev)),
            )
            if (dotp < 0).sum() != 0:
                project2cone2(
                    self.grads.to(dev)[:, id_task].unsqueeze(1),
                    self.grads.to(dev).index_select(1, indx.to(dev)),
                    self.reg,
                )
                # copy gradients back
                overwrite_grad(
                    self.model.parameters,
                    self.grads.to(dev)[:, id_task],
                    self.grad_dims,
                )

        elif self.CL == "L2" and not self.first_task:
            dev = next(self.model.parameters()).device
            l2_reg = 0

            for n, p in self.model.named_parameters():
                l = self.reg * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            self.log("l2_reg", l2_reg, on_epoch=True)
            loss = loss + l2_reg
        elif self.CL == "EWC" and not self.first_task:
            dev = next(self.model.parameters()).device
            ewc_loss = 0
            for n, p in self.model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                l = (
                    self.reg
                    * self.fisher[n].to(dev)
                    * (p - self.optpar[n].to(dev)).pow(2)
                )
                ewc_loss += l.sum()
            self.log("EWC_reg", ewc_loss, on_epoch=True)
            loss = loss + ewc_loss

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.CL == "ADAPTER":
            task_id = str(self.task_list_seen.index(batch["task_id"][0]))
            self.model.train_adapter(task_id)
            self.model.set_active_adapters(task_id)
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )[0]
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.CL == "ADAPTER":
            parameters_to_update = [
                p for n, p in self.named_parameters() if "adapter" in str(n)
            ]
            return AdamW(parameters_to_update, lr=self.lr, correct_bias=True)
        else:
            return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def backward(self, loss, optimizer, optimizer_idx):
        if (self.CL == "GEM" or self.CL == "AGEM") and not self.first_task:
            pass
        else:
            loss.backward()
