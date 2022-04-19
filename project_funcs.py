def update_adapters(task_id, task_loader, model):
    new_adapter_weights = find_best_merge(task_id, task_loader, model)
    # Modify the weights of the adapters based on what is returned
    active_state_dict = model.state_dict()

    # Change the current model to have adapter weights that were
    # selected by [find_best_merge]
    for layer_name, new_tensor_weight in new_adapter_weights.keys():
        active_state_dict[layer_name] = new_tensor_weight

    model = model.load_state_dict(active_state_dict)
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
    weights = torch.zeros(100)

    for lamb in range(0, 1, 0.1):
        lamb_2 = 1 - lamb
        # Merge and Score Here

    return score, weights
