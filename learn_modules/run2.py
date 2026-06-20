import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")

def activation(text, layer):
    ids = tok(text, return_tensors="pt").to(model.device)
    # hidden_states is a tuple: [embeddings, after-layer-1, after-layer-2, ...]
    hs = model(**ids, output_hidden_states=True).hidden_states
    return hs[layer][0, -1]   # the LAST token's vector at this layer

happy = [
    "I am so thrilled and joyful today!",
    "This is wonderful, I feel fantastic!",
    "What a delightful, cheerful morning!",
    "I'm overjoyed, everything is amazing!",
]
sad = [
    "I am so miserable and hopeless today.",
    "This is awful, I feel terrible.",
    "What a bleak, depressing morning.",
    "I'm devastated, everything is ruined.",
]

LAYER = 12
def mean_act(sentences):
    acts = [activation(s, LAYER) for s in sentences]   # each is [896]
    return torch.stack(acts).mean(dim=0)               # average -> [896]

happy_dir = mean_act(happy) - mean_act(sad)            # the "happiness direction"
print("direction shape:", happy_dir.shape)
print("direction norm :", happy_dir.norm().item())


##### TEST IT OUT


import torch.nn.functional as F

def score(text):
    a = activation(text, LAYER)
    return F.cosine_similarity(a, happy_dir, dim=0).item()

print("new happy :", score("I just got the best news, I'm elated!"))
print("new sad   :", score("I lost everything, I can't stop crying."))
print("neutral   :", score("The report is due on Tuesday afternoon."))