# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.autograd as autograd
# import torch.nn.functional as F

# import torchvision
# import numpy as np
# import math
# from tqdm import tqdm

# # Subnetwork forward from hidden networks
# class GetSubnet(autograd.Function):
#     @staticmethod
#     def forward(ctx, scores):
#         return (scores >= 0).float()

#     @staticmethod
#     def backward(ctx, g):
#         # send the gradient g straight-through on the backward pass.
#         return g


# def mask_init(module):
#     scores = torch.Tensor(module.weight.size())
#     nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
#     return scores


# def signed_constant(module):
#     fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
#     gain = nn.init.calculate_gain('relu')
#     std = gain / math.sqrt(fan)
#     module.weight.data = module.weight.data.sign() * std

# class MultitaskMaskLinear(nn.Linear):
#     def __init__(self, *args, num_tasks=1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_tasks = num_tasks
#         self.scores = nn.ParameterList(
#             [
#                 nn.Parameter(mask_init(self))
#                 for _ in range(num_tasks)
#             ]
#         )
        
#         # Keep weights untrained
#         self.weight.requires_grad = False
#         signed_constant(self)
    
#     @torch.no_grad()
#     def cache_masks(self):
#         self.register_buffer(
#             "stacked",
#             torch.stack(
#                 [
#                     GetSubnet.apply(self.scores[j])
#                     for j in range(self.num_tasks)
#                 ]
#             ),
#         )

#     def forward(self, x):
#         if self.task < 0:
#             # Superimposed forward pass
#             alpha_weights = self.alphas[: self.num_tasks_learned]
#             print(x.size())
#             input()
#             print(alpha_weights)
#             idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
#             print(idxs)
#             input()

#             if len(idxs.shape) == 0:
#                 idxs = idxs.view(1)
#             print(idxs)
#             input()
            
#             print(alpha_weights[idxs])
#             input()

#             print(self.stacked.size())
#             input()

#             print(self.stacked[: self.num_tasks_learned][idxs].size())
#             input()

#             subnet = (
#                 alpha_weights[idxs]
#                 * self.stacked[: self.num_tasks_learned][idxs]
#             ).sum(dim=0)
#             print(subnet.size())
#             print("END")
#             input()

#         else:
#             # Subnet forward pass (given task info in self.task)
#             subnet = GetSubnet.apply(self.scores[self.task])
#         w = self.weight * subnet
#         x = F.linear(x, w, self.bias)
#         return x


#     def __repr__(self):
#         return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"


# # Utility functions
# def set_model_task(model, task, verbose=True):
#     for n, m in model.named_modules():
#         if isinstance(m, MultitaskMaskLinear):
#             if verbose:
#                 print(f"=> Set task of {n} to {task}")
#             m.task = task

# def cache_masks(model):
#     for n, m in model.named_modules():
#         if isinstance(m, MultitaskMaskLinear):
#             print(f"=> Caching mask state for {n}")
#             m.cache_masks()

# def set_num_tasks_learned(model, num_tasks_learned):
#     for n, m in model.named_modules():
#         if isinstance(m, MultitaskMaskLinear):
#             print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
#             m.num_tasks_learned = num_tasks_learned

# def set_alphas(model, alphas, verbose=True):
#     for n, m in model.named_modules():
#         if isinstance(m, MultitaskMaskLinear):
#             if verbose:
#                 print(f"=> Setting alphas for {n}")
#             m.alphas = alphas

# # Multitask Model, a simple fully connected model in this case
# class MultitaskFC(nn.Module):
#     def __init__(self, hidden_size, num_tasks):
#         super().__init__()
#         self.model = nn.Sequential(
#             MultitaskMaskLinear(
#                 784,
#                 hidden_size,
#                 num_tasks=num_tasks,
#                 bias=False
#             ),
#             nn.ReLU(),
#             MultitaskMaskLinear(
#                 hidden_size,
#                 hidden_size,
#                 num_tasks=num_tasks,
#                 bias=False
#             ),
#             nn.ReLU(),
#             MultitaskMaskLinear(
#                 hidden_size,
#                 100,
#                 num_tasks=num_tasks,
#                 bias=False
#             )
#         )
    
#     def forward(self, x):
#         return self.model(x.flatten(1))


# class MNISTPerm:
#     class permute(object):
#         def __call__(self, tensor):
#             out = tensor.flatten()
#             out = out[self.perm]
#             return out.view(1, 28, 28)

#         def __repr__(self):
#             return self.__class__.__name__

#     def __init__(self, seed=0):
#         super(MNISTPerm, self).__init__()
        
#         data_root = "mnist"
#         self.permuter = self.permute()
#         self.seed = seed
#         train_dataset = torchvision.datasets.MNIST(
#             data_root,
#             train=True,
#             download=True,
#             transform=torchvision.transforms.Compose(
#                 [
#                     torchvision.transforms.ToTensor(),
#                     torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#                     self.permuter,
#                 ]
#             ),
#         )

#         # Data loading code
#         self.train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=128, shuffle=True
#         )
#         self.val_loader = torch.utils.data.DataLoader(
#             torchvision.datasets.MNIST(
#                 data_root,
#                 train=False,
#                 transform=torchvision.transforms.Compose(
#                     [
#                         torchvision.transforms.ToTensor(),
#                         torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#                         self.permuter,
#                     ]
#                 ),
#             ),
#             batch_size=128,
#             shuffle=False,
#         )

#     def update_task(self, i):
#         np.random.seed(i + self.seed)
#         self.permuter.__setattr__("perm", np.random.permutation(784))
    
#     def unpermute(self):
#         self.permuter.__setattr__("perm", np.arange(784))


# mnist = MNISTPerm()

# # Showing some example images from MNISTPerm
# mnist.unpermute()
# batch, labels = next(iter(mnist.val_loader))

# mnist.update_task(0)
# task0, labels = next(iter(mnist.val_loader))

# torchvision.transforms.ToPILImage()(
#     torchvision.utils.make_grid(
#         torch.cat([batch, task0], dim=-1)[:64],
#         normalize=True,
#         padding=5,
#         pad_value=0.2
#     )
# )


# # Finding supermasks per task

# def train(model, trainloader, optimizer, epoch):
#     model.train()

#     criterion = nn.CrossEntropyLoss()
#     num_correct = 0
#     total_seen = 0
#     for i, (batch, labels) in tqdm(
#         enumerate(trainloader),
#         ascii=True,
#         total=len(trainloader)
#     ):
#         logits = model(batch)
#         loss = criterion(logits, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if i % 20 == 0:
#             predictions = logits.argmax(dim=-1)
#             num_correct += (predictions == labels).float().sum()
#             total_seen += logits.size(0) 
#             tqdm.write(
#                 (f"e{epoch} {i+1}/{len(trainloader)}"
#                 f" => Loss {loss.item():0.4f}, "
#                 f"Acc@1 {(num_correct / total_seen):0.4f}"),
#                 end="\r"
#             )


# @torch.no_grad()
# def evaluate(model, val_loader, epoch):
#     model.eval()
#     num_correct = 0
#     total_seen = 0
#     for batch, labels in tqdm(
#         val_loader,
#         ascii=True,
#         total=len(val_loader)
#     ):
#         logits = model(batch)
#         predictions = logits.argmax(dim=-1)
#         num_correct += (predictions == labels).float().sum()
#         total_seen += logits.size(0) 
    

#     tqdm.write(
#         f"Val Perf after {epoch + 1} epochs "
#         f"Acc@1 {(num_correct / total_seen):0.4f}", 
#     )
#     return num_correct / total_seen


# num_tasks = 2 # For demonstration purposes, we go up to 2500 in our paper
# model = MultitaskFC(hidden_size=300, num_tasks=num_tasks)

# for task in range(num_tasks):
#     print(f"Training for task {task}")
#     set_model_task(model, task)
#     mnist.update_task(task)

#     optimizer = optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=1e-4)
#     # Train for 1 epoch
#     for e in range(1):
#         train(model, mnist.train_loader, optimizer, e)
        
#         print("Validation")
#         print("============")
#         acc1 = evaluate(model, mnist.val_loader, e)
        
    
#     cache_masks(model)
#     print()
#     set_num_tasks_learned(model, task + 1)
#     print()




# # When task info is not provided we can infer it
# # here we use the oneshot task inference alg detailed in the paper

# def oneshot_task_inference(model, batch, num_tasks):
#     # Set task < 0 for inference mode
#     set_model_task(model, -1, verbose=False)
    
#     # Initialize alphas to uniform
#     alphas = torch.ones(num_tasks, 1, 1) / num_tasks
#     print(alphas)
#     print(alphas.size())
#     alphas.requires_grad_(True)
#     set_alphas(model, alphas, verbose=False)
    
#     logits = model(batch)
    
#     # Entropy of logits
#     entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()
    
#     # Gradient wrt alphas
#     g, = autograd.grad(entropy, alphas)
#     print(g)
#     inferred_task = (-g).squeeze().argmax()

#     return inferred_task.item()


# num_examples = 1
# trials = 50
# num_correct = 0
# num_seen = 0
# for _ in tqdm(range(trials)):
#     for task in range(5):
#         mnist.update_task(task)
#         inferred_task = oneshot_task_inference(
#             model,
#             batch=next(iter(mnist.val_loader))[0][:num_examples],
#             num_tasks=5
#         )
#         if inferred_task == task:
#             num_correct += 1
#         num_seen += 1

# print(f"Task inference accuracy: {100 * num_correct / num_seen}%")

import glob
import re
import os
 

def get_checkpoint(log_dir, index_to_load):
    file = glob.glob(f"{log_dir}/*")
    for f in file:
        f_noprefix = f.replace(f"{log_dir}","")
        num = [int(s) for s in re.findall(r'\d+', f_noprefix)]
        if index_to_load in num:
            version = os.listdir(f+"/lightning_logs")[0]
            check_name = os.listdir(f+"/lightning_logs/"+ version+"/checkpoints/")[0]
            checkpoint_name = f.replace("[","\[").replace("]","\]").replace("\'","\\'")+"/lightning_logs/"+ version+"/checkpoints/"+check_name
    return checkpoint_name


log_dir = "runs_E2E/BEST/ADAPTER_EPC_10_LR_0.000625_BOTL_300__gpt2"
index_to_load = 13
get_checkpoint(log_dir, index_to_load)
