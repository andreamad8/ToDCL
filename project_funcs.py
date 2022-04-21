def update_adapters(task_id, task_loader, model):
    new_adapter_weights = find_best_merge(task_id, task_loader, model)
    # Modify the weights of the adapters based on what is returned
    active_state_dict = model.state_dict()

    # Change the current model to have adapter weights that were
    # selected by [find_best_merge]
    for layer_name, new_tensor_weight in new_adapter_weights.keys():
        active_state_dict[layer_name] = new_tensor_weight

    model.load_state_dict(active_state_dict)
    return model


def find_best_merge(current_task_id, task_loader, model):
    # I am expecting best weights to simply be
    # a dictionary of named layer to the tensor of weights/bias
    # The named layer parameter e.g. ("model.encoder.block.0.layer.0.adapters.3.adapter_up.weight" : weight tensor)
    #
    # This should be able to be found when merging as in order to merge you will probably
    # want to iterate through model.parameters() which will give you name,tensor
    # In the code they check if ["adapter" in layer_name] to determine
    # if this is an adapter layer
    best_weights = dict()

    # Iterate Through and Find Best Merge
    for task_id in model.task_list_seen:
        score, weights = score_merge(current_task_id, task_id, task_loader, model)

    return best_weights


def score_merge(current_task_id, task_id, task_loader, model):
    score = None

    fisher_source, param_source = compute_fisher(task_loader, model, task_id)
    fisher_target, param_target = compute_fisher(task_loader, model, current_task_id)

    model.train_adapter("temporary")
    model.set_active_adapters("temporary")

    best_score = float("-inf")
    best_weights = None
    for lamb in range(0, 1, 0.1):
        weights = set_params(
            model,
            fisher_source,
            param_source,
            fisher_target,
            param_target,
            lamb,
            current_task_id,
        )
        score = evaluate(task_loader, model)
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_score, best_weights


def evaluate(task_loader, model):
    model.eval()
    loss = 0
    with torch.no_grad():
        for idx, inputs in enumerate(task_loader):
            loss -= self.model(
                input_ids=inputs["encoder_input"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["decoder_output"],
            )[0]

    return loss


def set_params(
    model, fisher_source, param_source, fisher_target, param_target, lamb, target_id
):
    param_dict = {}
    lamb_2 = 1 - lamb
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix, suffix = name.split(".temporary.")
            name = prefix + suffix
            source_fish = fisher_source[name]
            target_fish = fisher_target[name]
            source = param_source[name]
            target = param_target[name]
            reg = (lamb * source_fish) + (lamb_2 * target_fish)
            # Default to Target if Fisher is small
            if reg < 1e-8:
                merge = target
            else:
                merge = (
                    (lamb * source_fish * source) + (lamb_2 * target_fish * target)
                ) / reg
            param.copy_(merge)
            param_dict[prefix + "." + target_id + "." + suffix] = merge
    return param_dict


def compute_fisher(task_loader, model, task_id, num_samples=1024):
    model.train_adapter(task_id)
    model.set_active_adapters(task_id)
    gradients_dict = {}
    param_dict = {}

    for name, param in model.named_parameters():
        # Only Compute For Active Adapter
        if param.requires_grad:
            prefix, suffix = name.split("." + task_id + ".")
            name = prefix + suffix
            param_dict[name] = param
            gradients_dict[name] = torch.zeros_like(param)

    for idx, inputs in enumerate(task_loader):
        if idx >= num_samples:
            break

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v

        loss = self.model(
            input_ids=inputs["encoder_input"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["decoder_output"],
        )[0]

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                prefix, suffix = name.split("." + task_id + ".")
                name = prefix + suffix
                gradients_dict[name] += torch.square(param.grad).data

        model.zero_grad()

    return gradients_dict, param_dict
