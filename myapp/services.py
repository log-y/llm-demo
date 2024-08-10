import torch
import os
from .models.model import GPT
from .models.model import GPTConfig
import tiktoken
from torch.nn import functional as F
import random

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pt')
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

model = GPT(GPTConfig(vocab_size=50257))
model.load_state_dict(checkpoint)

model.eval()
enc = tiktoken.get_encoding("gpt2")
device="cpu"

sample_rng = torch.Generator(device=device)

def generate_text(text, num_predictions=10):
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(1, 1)
    xgen = tokens.to(device)
    sample_rng.manual_seed(random.randint(20,900))
    with torch.no_grad():
        logits, _ = model(xgen)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, num_predictions, dim=-1)
    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
    xcol = torch.gather(topk_indices, -1, ix)
    xgen = torch.cat((xgen, xcol), dim=1)

    tokenprobs = [(enc.decode_single_token_bytes(topk_indices[0, i].item()).decode('utf-8'), topk_probs[0, i].item()) for i in range(num_predictions)]

    newtoken = xcol[0,0].item()
    newtokenstring = enc.decode_single_token_bytes(newtoken).decode('utf-8')

    return newtokenstring, tokenprobs
        
    