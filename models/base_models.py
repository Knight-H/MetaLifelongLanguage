import math
import random
import os
from collections import defaultdict

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_utils import ModuleUtilsMixin, no_init_weights
from transformers.utils import hf_bucket_url, cached_path

from typing import Optional, Tuple, Union
import numpy as np


class TransformerClsModel(nn.Module):
    def __init__(self, model_name, n_classes, max_length, device):
        super(TransformerClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.encoder = GPT2Model.from_pretrained('gpt2')
        else:
            raise NotImplementedError
        self.linear = nn.Linear(768, n_classes)
        self.to(self.device)
        
        # Custom additions from LAMOL
#         self.special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
#         self.tokenizer.add_tokens(list(special_tokens.values()))
#         self.special_token_ids = {k:self.tokenizer.convert_tokens_to_ids(v) for k,v in self.special_tokens.items()}


    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, out_from='full'):
        if out_from == 'full':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            out = self.linear(out)
        elif out_from == 'transformers':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        elif out_from == 'linear':
            out = self.linear(inputs)
        else:
            raise ValueError('Invalid value of argument')
        return out


class TransformerRLN(nn.Module):

    def __init__(self, model_name, max_length, device, token_weight=5):
        super(TransformerRLN, self).__init__()
        self.max_length = max_length
        self.device = device
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.encoder = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model_config = GPT2Config.from_pretrained('gpt2')
        else:
            raise NotImplementedError
        self.to(self.device)
        
        # Custom additions from LAMOL
#         self.special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
#         self.tokenizer.add_tokens(list(special_tokens.values()))
#         self.special_token_ids = {k:self.tokenizer.convert_tokens_to_ids(v) for k,v in self.special_tokens.items()}
        
#         self.model_config.vocab_size = len(self.tokenizer)
#         self.tokens_weight = torch.ones([self.model_config.vocab_size], dtype=torch.float).to(self.device)
#         self.tokens_weight[self.special_token_ids["ans_token"]] = tokens_weight  # only answer token has token weight of 5! (default)
        

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return out

# Copying class GPT2Model(GPT2PreTrainedModel)
#      https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L667
# class GPT2PreTrainedModel(PreTrainedModel):
#      https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L439
# class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
#      https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L842

# Copying class GPT2LMHeadModel(GPT2PreTrainedModel)
# Copied from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py#L668
# Class GPT2Model
class CustomGPT2RLN(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]
    
    def __init__(self, model_name = 'gpt2', max_length=1024, device='cpu', token_weight=5):
        # Fix the damn config!
        config = GPT2Config.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.ans_token_weight = token_weight
        
        #### Start Custom Code ####
        
        # Number of Hidden Layers -> 12 to 11
        # And move layernorm!!
        config.n_layer = 11
        #### End Custom Code ####
        
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    #### Bring special token addition here since it must be loaded first (weights) ####
    def add_special_tokens_init(self, device='cpu'):
        # Custom additions from LAMOL
        self.special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
        self.tokenizer.add_tokens(list(self.special_tokens.values()))
        self.special_token_ids = {k:self.tokenizer.convert_tokens_to_ids(v) for k,v in self.special_tokens.items()}
        
        self.config.vocab_size = len(self.tokenizer)
        self.tokens_weight = torch.ones([self.config.vocab_size], dtype=torch.float).to(device)
        self.tokens_weight[self.special_token_ids["ans_token"]] = self.ans_token_weight  # only answer token has token weight of 5! (default)
        
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # input_ids should be [10,23,4029, 3920] ... a bunch of tokenized words
    # so input_shape is just input_ids.size() ~ (1024,) [length]
    # output_shape = input_shape + (hidden_states.size(-1),) where prev hidden_states i inputs_embeds 
    #    >> this means that output_shape ~(1024, 768, )  ?? default "n_embd": 768,
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # Get the config/ specified values in forward
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #  Get the input_ids/input_embeds
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
            
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
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
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Size before in torch.Size([1, len, 768])
            #print("SIZE BEFORE IN >> ", hidden_states.size())

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

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        
        #print(hidden_states.size())
        hidden_states = hidden_states.view(output_shape) # Changes from torch.Size([1, len, 768]) -> torch.Size([len, 768])
        #print(hidden_states.size())
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    

    
# Copied from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py#L946
# class GPT2Model + GPT2LMHeadModel
# This Requires ModuleUtilsMixin to use get_head_mask
class CustomGPT2LMHeadPLN(nn.Module, ModuleUtilsMixin):
    
    # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py#L452
    base_model_prefix = "transformer"
    
    def __init__(self, config):
        super(CustomGPT2LMHeadPLN, self).__init__()
        self.config = config
        
        # Last H Layer
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=11)])
        # Last LN-F layer
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Need to tie this to embedding model!!
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    # From PretrainedModel https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L1396
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
                        config, input_embeddings, *model_args, **model_kwargs):
        WEIGHTS_NAME = "pytorch_model.bin"

        model_name = 'gpt2'
        filename = WEIGHTS_NAME
        revision = None
        mirror = None
        cache_dir = None
        force_download = False
        proxies = None
        resume_download = False
        local_files_only = False
        use_auth_token = None
        from_auto_class = False
        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        from_pt = True
        _fast_init = True

        archive_file = hf_bucket_url(
            model_name,
            filename=filename,
            revision=revision,
            mirror=mirror,
        )
        print("URL: ", archive_file)

        # Load from URL or cache if already cached
        resolved_archive_file = cached_path(
            archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
        )

        print("CACHED: ", resolved_archive_file)

        # def load_state_dict  https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L344
        state_dict = torch.load(resolved_archive_file,  map_location="cpu")

        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype_orig)

        with no_init_weights(_enable=_fast_init):
            model = cls(config, *model_args, **model_kwargs)
        
        # Load Pretrained Model here https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L1882
        # def _load_pretrained_model
        # But what I need is simple, just load h.11.x from the original model to h.0.x in the new model
        #  ln_f.weight, ln_f.bias will need to be loaded
        # lm_head.weight will need to be tied to the input embeddings.
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        loaded_keys = list(state_dict.keys())
        prefix = model.base_model_prefix
        
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L1974
        # Make sure we are able to load base models as well as derived models (with heads)
        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        start_prefix = ""
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are training to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
        
        # My Code
        keys_to_keep = ['ln_f.weight', 'ln_f.bias'] + [key for key in loaded_keys if key.startswith('h.11')]
        
        # Rename all state_dicts with h.x, and remove all unneeded keys
        for key in loaded_keys:
            if key in keys_to_keep:
                # If it is h, rename number in the middle to x -nlayer  (10-> 0, 11->1 if nlayer 10)
                # Rename similar to https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L385-L386
                if key.startswith('h.'):
                    prev_index = int(key.split('.')[1])
                    new_index = prev_index - config.n_layer
                    new_key = f"h.{new_index}.{'.'.join(key.split('.')[2:])}"
                    state_dict[new_key] = state_dict.pop(key)
                    print(f"key: {key} >> {new_key}")
            else:
                del state_dict[key]
        
        # Finally load the state Dict
        # def _load_state_dict_into_model https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L372
        # Didn't use this cuz too complicated!

        # Very similar to the way pyTorch writes load_state_dict 
        #   https://github.com/pytorch/pytorch/blob/v1.3.0/torch/nn/modules/module.py#L810-L824
        #model.load_state_dict(state_dict)
        
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []
        
        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, [], [], error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix=start_prefix)
        
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
            
        # Continue with Pretrained 
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py#L1893
        # make sure token embedding weights are still tied if needed
        model.tie_weights(input_embeddings) # Need to put in input_embedding since it's from another class!

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        
        return model
    
    def tie_weights(self, input_embeddings):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(output_embeddings, input_embeddings)

        # UNUSED!! if used will error cuz didnt copy tie encoder yet
        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)
        
        # Unused. but meh, leave it be for flexibility
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
                
                
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

        
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
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Just make the the same as the size that inputs in, they should know from last hidden state!!
        output_shape = input_ids.size()
        
        # the input of this is one of the last hidden states!
        # Need to change from torch.Size([len, 768]) -> torch.Size([1, len, 768])
        hidden_states = input_ids
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, 1) # I changed from config.h_layers to 1 since we only have 1!!
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
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
            
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    
class LinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(LinearPLN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, input):
        out = self.linear(input)
        return out


class TransformerNeuromodulator(nn.Module):

    def __init__(self, model_name, device):
        super(TransformerNeuromodulator, self).__init__()
        self.device = device
        if model_name == 'albert':
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768),
                                    nn.Sigmoid())
        self.to(self.device)

    def forward(self, inputs, out_from='full'):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out


class ReplayMemory:

    def __init__(self, write_prob, tuple_size):
        self.buffer = []
        self.write_prob = write_prob
        self.tuple_size = tuple_size

    def write(self, input_tuple):
        if random.random() < self.write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            self.write(write_tuple)

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []
