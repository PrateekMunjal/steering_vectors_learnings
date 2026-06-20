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

v = activation("The cat sat on the mat.", layer=12)
print("shape:", v.shape)       # how many numbers in one activation
print("first 5 values:", v[:5])


################

import torch.nn.functional as F

a = activation("I love this, it's absolutely wonderful!", 12)
b = activation("This is fantastic and makes me so happy!", 12)
c = activation("The mitochondria is the powerhouse of the cell.", 12)

print("happy vs happy :", F.cosine_similarity(a, b, dim=0).item())
print("happy vs neutral:", F.cosine_similarity(a, c, dim=0).item())

