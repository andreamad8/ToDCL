import os
import json
import torch
import numpy
import logging
import random
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from collections import defaultdict
from utils.dataloader import make_loader


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def get_example_inputs(model, tokenizer, prompt_text, device):
    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd
    num_layer = model.config.n_layer
    tokenizer.padding_side = "left"
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.float32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [
        2,
        batch_size,
        num_attention_heads,
        0,
        hidden_size // num_attention_heads,
    ]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return (
        input_ids.to(device),
        attention_mask.to(device),
        position_ids.to(device),
        empty_past,
    )


def test_generation_GPT2BATCH(
    model,
    tokenizer,
    input_text,
    device,
    do_sample=False,
    temperature=1.0,
    top_k=0,
    top_p=0,
    max_length=30,
    task_id=-1,
):
    eos_token_id = tokenizer.eos_token_id

    input_ids, attention_mask, position_ids, past = get_example_inputs(
        model, tokenizer, input_text, device
    )
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool).to(device)

    all_token_ids = input_ids.clone()
    task_id = str(task_id)
    for step in range(max_length):

        if task_id == -1:
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
            )
        else:
            model.train_adapter(task_id)
            model.set_active_adapters(task_id)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
            )

        next_token_logits = outputs[0][:, -1, :]

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1
        ).to(device)

        past = list(outputs[1])  # past in torch output is tuple
        if torch.all(has_eos):
            break

    responses = []
    responses_plain = []
    for i, output in enumerate(all_token_ids):
        responses_plain.append(tokenizer.decode(output, skip_special_tokens=True))
        res = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(res[res.find("[SOS]") :].replace("[SOS]", "").strip())
    return responses, responses_plain


def generate_sample_prev_task(
    args,
    model,
    tokenizer,
    dataset_dic,
    task_id_so_far,
    number_of_sample,
    time,
    task_id_adpt=-1,
):
    # device = torch.device(f"cuda:{args.GPU[0]}")
    device = "cpu"
    if (torch.cuda.is_available()):
        device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()
    ## notice that this sample is just to have the data struct
    temp_aug_mem = random.sample(
        dataset_dic["['sgd_restaurants']"],
        min(len(dataset_dic["['sgd_restaurants']"]), number_of_sample),
    )
    temp_aug_sam = random.sample(
        dataset_dic["['sgd_restaurants']"],
        min(len(dataset_dic["['sgd_restaurants']"]), number_of_sample),
    )
    with torch.no_grad():
        if "gpt2" in args.model_checkpoint:  ## this works only with GPT2
            sample_list = []
            for i in range(int(number_of_sample / (args.valid_batch_size)) + 1):
                if (
                    i % 2 == 0 or args.task_type != "E2E"
                ):  # sample on batch with and one without API call
                    input_batch = [
                        f"[{str(eval(task_id_so_far)[0])}]"
                        for _ in range(args.valid_batch_size)
                    ]
                else:
                    input_batch = [
                        f"[{str(eval(task_id_so_far)[0])}-API]"
                        for _ in range(args.valid_batch_size)
                    ]
                _, samples = test_generation_GPT2BATCH(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_batch,
                    device=device,
                    max_length=300,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.1,
                    task_id=task_id_adpt,
                )
                sample_list += samples
    sample_list = random.sample(sample_list, min(len(sample_list), number_of_sample))
    # this sample is to train the previous task generator
    for i in range(len(temp_aug_mem)):
        temp_aug_mem[i][
            "history_reply"
        ] = f"{sample_list[i].strip()} {tokenizer.eos_token}"
    # this sample is to train the previous task itself
    # hence we remove the special token in input
    for i in range(len(temp_aug_sam)):
        samp = sample_list[i].strip()
        samp = samp.replace(f"[{str(eval(task_id_so_far)[0])}]", "")
        samp = samp.replace(f"[{str(eval(task_id_so_far)[0])}-API]", "")
        temp_aug_sam[i]["history_reply"] = f"{samp} {tokenizer.eos_token}"

    temp_aug = temp_aug_mem + temp_aug_sam
    ## save the generated data for logging
    if not os.path.exists(f"{args.saving_dir}/{time}"):
        os.makedirs(f"{args.saving_dir}/{time}")
    with open(
        f"{args.saving_dir}/{time}" + f"/{task_id_so_far}_generated.json", "w"
    ) as fp:
        json.dump(temp_aug, fp, indent=4)
    return temp_aug


def test_model_seq2seq(args, model, tokenizer, test_loader, time="0_['']"):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()
    results = []

    for idx_b, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            if "gpt2" in args.model_checkpoint:
                value_batch, _ = test_generation_GPT2BATCH(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=[b + "[SOS]" for b in batch["history"]],
                    device=device,
                    max_length=100,
                )
            else:
                responses = model.generate(
                    input_ids=batch["encoder_input"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=100,
                )
                value_batch = tokenizer.batch_decode(
                    responses, skip_special_tokens=True
                )
        for idx, resp in enumerate(value_batch):
            results.append(
                {
                    "id": batch["dial_id"][idx],
                    "turn_id": batch["turn_id"][idx],
                    "dataset": batch["dataset"][idx],
                    "task_id": batch["task_id"][idx],
                    "spk": batch["spk"][idx],
                    "gold": batch["reply"][idx],
                    "genr": resp,
                    "hist": batch["history"][idx],
                }
            )
        # if(idx_b==1): break
    if not os.path.exists(f"{args.saving_dir}/{time}"):
        os.makedirs(f"{args.saving_dir}/{time}")
    with open(f"{args.saving_dir}/{time}" + "/generated_responses.json", "w") as fp:
        json.dump(results, fp, indent=4)
    tokenizer.padding_side = "right"


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


def test_model_seq2seq_ADAPTER(
    args, model, tokenizer, test_loader, test_dataset, time="0_['']", max_seen_task=0
):
    # device = torch.device(f"cuda:{args.GPU[0]}")
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    model.model.to(device)
    model.model.eval()
    results = []

    print(model.task_list_seen, len(model.task_list_seen))
    range_adpt = len(model.task_list_seen)

    perplexity_dict = {
        f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}': []
        for sample in test_dataset
    }
    for t in range(range_adpt):
        print(f"Task {t}")
        for idx_b, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            ppl_batch = model.compute_PPL(
                batch, task_id=t, device=device, tokenizer=tokenizer
            )  ## one value per batch
            for (d_id, t_id, ta_id, ppl) in zip(
                batch["dial_id"], batch["turn_id"], batch["task_id"], ppl_batch
            ):
                perplexity_dict[f"{d_id}_{t_id}_{ta_id}"].append(ppl)

    # select the task id with the lowest perplexity (loss)
    perplexity_dict_ = {}
    for k, v in perplexity_dict.items():
        if len(v) == range_adpt:
            perplexity_dict_[k] = v
        else:
            print(k, v)

    perplexity_dict = {k: argmin(v) for k, v in perplexity_dict_.items()}

    ## group by sample by predicted task id
    test_dataset_by_predicted_id = defaultdict(list)
    for sample in test_dataset:
        if (
            f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}'
            in perplexity_dict
        ):
            test_dataset_by_predicted_id[
                perplexity_dict[
                    f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}'
                ]
            ].append(sample)

    for k, v in test_dataset_by_predicted_id.items():
        print(f"Task {k}: {len(v)}")

    ## create a dataloader for batch each of this
    test_dataset_by_predicted_id = {
        k: make_loader(args, v, model.tokenizer)
        for k, v in test_dataset_by_predicted_id.items()
    }

    for pred_task_id, task_loader in tqdm(
        test_dataset_by_predicted_id.items(), total=len(test_dataset_by_predicted_id)
    ):
        pred_task_id = str(pred_task_id)
        # print(f"Task Id: {task_id}")
        for idx_b, batch in tqdm(enumerate(task_loader), total=len(task_loader)):
            with torch.no_grad():
                if "gpt2" in args.model_checkpoint:
                    value_batch, _ = test_generation_GPT2BATCH(
                        model=model,
                        tokenizer=tokenizer,
                        input_text=[b + "[SOS]" for b in batch["history"]],
                        device=device,
                        max_length=100,
                        task_id=pred_task_id,
                    )
                else:
                    model.model.set_active_adapters(task_id)
                    responses = model.model.generate(
                        input_ids=batch["encoder_input"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=100,
                    )
                    value_batch = tokenizer.batch_decode(
                        responses, skip_special_tokens=True
                    )
            for idx, resp in enumerate(value_batch):
                results.append(
                    {
                        "id": batch["dial_id"][idx],
                        "turn_id": batch["turn_id"][idx],
                        "dataset": batch["dataset"][idx],
                        "task_id": batch["task_id"][idx],
                        "spk": batch["spk"][idx],
                        "gold": batch["reply"][idx],
                        "genr": resp,
                        "hist": batch["history"][idx],
                        "pred_task_id": pred_task_id,
                    }
                )

            # if(idx_b==1): break
    if not os.path.exists(f"{args.saving_dir}/{time}"):
        os.makedirs(f"{args.saving_dir}/{time}")
    with open(f"{args.saving_dir}/{time}" + "/generated_responses.json", "w") as fp:
        json.dump(results, fp, indent=4)
    tokenizer.padding_side = "right"


# def test_model(args,model,tokenizer,test_loader,time=0):
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(args.device)
# model.eval()
# results = []
# for task_id, task_loader in tqdm(test_loader.items(),total=len(test_loader)):
#     # print(f"Task Id: {task_id}")
#     for idx_b, batch in enumerate(task_loader):
#         input_ids, _, token_type_ids  = tuple(torch.tensor([batch[input_name]]).to(args.device) for input_name in MODEL_INPUTS)
#         with torch.no_grad():
#             response = generate(args,model,tokenizer,input_ids,token_type_ids)
#         results.append({"id":batch["dial_id"],"turn_id":batch["turn_id"],
#                         "dataset":batch["dataset"],"task_id":task_id,
#                         "spk":batch["spk"],"gold":batch["row_reply"],
#                         "genr":response,"hist":batch["plain_history"]})

# if not os.path.exists(f'{args.saving_dir}/{time}'):
#     os.makedirs(f'{args.saving_dir}/{time}')
# with open(f'{args.saving_dir}/{time}'+'/generated_responses.json', 'w') as fp:
#     json.dump(results, fp, indent=4)


def test():
    pass
    # args = get_args()
    # model = Seq2SeqToD(args)
    # model.model.load_state_dict(torch.load(f'runs_INTENT/BEST/ADAPTER_EPC_10_LR_0.00625_BOTL_100__gpt2/'))

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _, _, test_loader, (_, _) = get_data_loaders(args, tokenizer, test=True)
    # train_loader, val_loader, dev_val_loader, (train_datasets, test_datasets) = get_data_loaders(args, model.tokenizer)

    # print(f"Loading Model: {args.model_checkpoint}")
    # model.to(args.device)
    # test_model(args,model,tokenizer,test_loader)


if __name__ == "__main__":
    test()
