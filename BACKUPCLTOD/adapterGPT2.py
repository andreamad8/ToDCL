
from transformers import GPT2Model, GPT2Tokenizer, GPT2PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def from_polar(r, phi):
    a = r*torch.cos(phi)
    b = r*torch.sin(phi)
    return a, b


def to_polar(v):
    r = torch.norm(v, p=2, dim=0)
    phi = torch.atan2(v[1], v[0])
    return r, phi


def cmul(k, v):
    a = k[0]*v[0] - k[1]*v[1]
    b = k[1]*v[0] + k[0]*v[1]
    return torch.stack([a, b])


def cdiv(k, v):
    a = k[0]*v[0] + k[1]*v[1]
    b = k[1]*v[0] - k[0]*v[1]
    return torch.stack([a, b])/(v[0]**2 + v[1]**2)


class ComplexVar(object):
    def __init__(self, a, b, polar_init=False):
        self.s_r = torch.FloatTensor(*((2,) + a.shape))
        if polar_init:
            a, b = from_polar(a, b)

        self.s_r[0] = a
        self.s_r[1] = b

    def to_polar(self):
        return to_polar(self.s_r) 


class Adapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(Adapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.project_down = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        self.project_up = nn.Linear(bottleneck, nx)

    def forward(self, x, rotation=None,superpostion=False):
        if(superpostion):
            x = x*rotation
        x_ = self.ln(x)
        x_ = self.project_down(x_)
        x_ = self.relu(x_)
        x_ = self.project_up(x_)
        x  = x + x_ #residual connection
        return x

class ComplexLinear(nn.Module):
    '''Complex layer'''
    def __init__(self, n_in, n_out):
        super(ComplexLinear, self).__init__()
        a = torch.empty(n_in, n_out)
        a = nn.init.xavier_normal_(a)
        b = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi) 
        self.cv = ComplexVar(a, b, polar_init=True)
        self.w = nn.Parameter(self.cv.s_r)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x_a, x_b):
        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(x_a, w_a) - torch.mm(x_b, w_b)
        r_b = torch.mm(x_b, w_a) + torch.mm(x_a, w_b)
        return r_a + self.bias, r_b

class AdapterComplex(nn.Module):
    def __init__(self, config, bottleneck):
        super(Adapter1, self).__init__()
        nx = config.n_embd
        self.w = ComplexLinear(nx, nx)
        self.relu = nn.ReLU()

    def forward(self, x, rotation=None,superpostion=False):
        return self.w(x*rotation) + x*rotation


class FAdapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(FAdapter, self).__init__()
        self.nx = config.n_embd
        self.eps = config.layer_norm_epsilon

    def forward(self, x, W_up, b_up, W_down, b_down, W_ln,b_ln):
        normalized_shape = x.size()[1:]
        x_ = F.layer_norm(x, normalized_shape=normalized_shape, 
                             weight=W_ln.expand(normalized_shape), 
                             bias=b_ln.expand(normalized_shape), 
                             eps=self.eps)
        x_ = F.linear(x_, weight=W_down, bias=None) 
        x_ = F.relu(x_)
        x_ = F.linear(x_, weight=W_up, bias=None) 
        x  = x + x_ #residual connection
        return x

class SupAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=400, adapter_num=25, complex=False):
        super(SupAdapter, self).__init__()
        nx = config.n_embd
        if complex:
            self.mixadapter = nn.ModuleList([AdapterComplex(config, bottleneck_size)])
        else:
            self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size)])
            ## rotation ==> this is equivalent to scipy.linalg.hadamard
            rand_01 = np.random.binomial(p=.5, n=1, size=(nx, adapter_num)).astype(np.float32)
            o = torch.from_numpy(rand_01*2 - 1)
            self.o = nn.Parameter(o)

    def forward(self, x, task_id=-1, alphas=None, current_task_id=None, complex=False):
        if task_id==-1:
            return x
        else:
            return self.mixadapter[0](x,self.o[:,task_id],True)


class MixAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=400, adapter_num=25):
        super(MixAdapter, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size) for _ in range(adapter_num)])
        self.func_adapter = FAdapter(config, bottleneck_size)
    
    def get_zero_weight(self):
        di = {}
        for n, p in self.mixadapter[0].named_parameters():
            di[n] = torch.zeros(p.size()).to(p.get_device()) 
        return di 

    def superipose_paramters(self, x, alpha, current_task_id):
        base_weight = self.get_zero_weight()
        for id_adpt,adpt in enumerate(self.mixadapter[:current_task_id]):
            for n, p in adpt.named_parameters():
                base_weight[n] += alpha[id_adpt]*p

        return self.func_adapter(x=x, 
                                 W_up=base_weight['project_up.weight'], 
                                 b_up=base_weight['project_up.bias'], 
                                 W_down=base_weight['project_down.weight'], 
                                 b_down=base_weight['project_down.bias'], 
                                 W_ln=base_weight['ln.weight'],
                                 b_ln=base_weight['ln.bias'])


    def forward(self, x, task_id=-1, alphas=None, current_task_id=-1):
        if task_id==-1:
            return x
        elif task_id == -2:
            return self.superipose_paramters(x, alphas, current_task_id)
        else:
            return self.mixadapter[task_id](x)


class GPT2Adapter(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.config = config
        
    def get_output_embeddings(self):
        return self.lm_head

    def add_adapters(self,bottleneck_size=100,adapter_num=40,superposition=False):
        if(superposition):
            self.adapter_blocks = nn.ModuleList([SupAdapter(self.config,bottleneck_size,adapter_num) for _ in range(self.config.n_layer)])
        else:
            self.adapter_blocks = nn.ModuleList([MixAdapter(self.config,bottleneck_size,adapter_num) for _ in range(self.config.n_layer)])

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id = -1,
            alphas=None,
            current_task_id=None,
            complex=False,
            **kwargs,
        ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        # transformer_outputs = self.transformer(
        #     input_ids,
        #     past_key_values=past_key_values,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        ### OVERLOADING THIS FUNCTION 
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.transformer.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.transformer.config.use_cache
        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.transformer.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.transformer.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.transformer.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.transformer.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if(complex):
            img_part = torch.zeros_like(hidden_states)
        for i, (block, layer_past, adapter) in enumerate(zip(self.transformer.h, past_key_values, self.adapter_blocks)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.transformer.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            if(complex):
                outputs[0], img_part = adapter((outputs[0],img_part), task_id=task_id, alphas=alphas, current_task_id=current_task_id,complex=complex)
            else:
                outputs[0] = adapter(outputs[0], task_id=task_id, alphas=alphas, current_task_id=current_task_id)

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            transformer_outputs = tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)
        else:
            transformer_outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )


        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

