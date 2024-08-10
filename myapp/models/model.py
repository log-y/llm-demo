import tiktoken
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os

#This represents the self-attention stage, responsible for understanding how
#each token should 'attend' to other tokens. This is required to extract how tokens
#should interact gramatically and extracts meaning from the text. Multiple heads
#allow for multiple understandings.
class CausalSelfAttention(nn.Module):
    #sets up learnable parameters and a buffer
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 #to ensure that the embedding dimension can be split into multiple heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #create 'learnable' weight embeddings for kqv matrices
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) #

        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        #above 3 lines set up a buffer for the attention mask

    def forward(self, x):
        B, T, C = x.size() #extracts dimensions of input tensor 'x'
        qkv = self.c_attn(x) #apply the linear transformation (by the learnerd weight matrix)
        q, k, v = qkv.split(self.n_embd, dim=2) #extract qkv matrices
        #reshape and transposed for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        #performed scaled dot product attention, heavy matrix multiplication so you use a
        #pytorch function called FlashAttention. it saves the NxN attention matrix to HBM
        #to prevent repeated access. this is a notable optimization
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #another reshaping, and projection back to embedding dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

#a feedforward network designed to let the model learn non-linear relationships
#about its tokens
class MLP(nn.Module):
    #sets up two linear layers that use GELU
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) #transposes to higher dimension
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) #reduction back to original dimension
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#a 'layer' in the transformer architecture. houses the different stages in the
#transformer process.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) #normalization for model stability
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length in tokens
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    #when changing the above parameters, stick to powers of 2, or 4, or 8.
    #this is because matrix multiplication is optimized for these numbers

#overhead control of the layers
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #declares an entire transformer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #holds word embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), #holds positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #holds the layers
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight typing scheme, makes them 'share' the same matrix (for optimization)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights) #applies custom weight initialization
    #initialize the weights in the beginning of model training, establishes
    #a 'range' of values a weight can fall into. here, all of the weights will
    #be random numbers with a mean of 0 and a stdev of 0.02
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * module.NANOGPT_SCALE_INIT) ** -0.5 #scales down stdev
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) #could change std based on size of model
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #defines a forward pass of some data (training or validation)
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb #add positional embeddings to token
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size), one last linear layer
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            #flats out 3-d into 2d, also flats out targets into 2d, pass into cross_entropy to calculate a loss
            #uses cross entropy formula to calculate loss
        return logits, loss



    #provides a function to load from previously trained model (gpt2)
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params, 25 is ugly num
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    #sets up weight decay optimization for the optimizer.
    #weight decay adds a penalization score to larger weights in the parameters.
    #this is useful to prevent overfitting (helps with generalization).
    #also checks for fused adamW, which combines multiple operations into
    #a single kernel for optimization. (GPU-only)
    def configure_optimizers(self, weight_decay, learning_rate, device):
        #start with candidate parameters requiring grad, basically create
        #a dictionary of parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #decay parameters (parameters with 2+ dimensions, typically in linear layer)
        #you only want weight decay in 2+ layered weights because those layers
        #have lots of parameters, which can easily lead to overfitting (learning data noise)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        #no-decay parameters (parameters with 1 dimension, typically a bias)
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        #make two parameter groups. one with weight decay and one without
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        #calculates and prints the number of parameters in each group
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        #check if fused AdamW implementation is available, then uses it
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}") #introduce fused adam to combine calculations over different kernels to reduce overhead
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer