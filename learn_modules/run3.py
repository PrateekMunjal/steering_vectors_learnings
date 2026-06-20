import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")

LAYER = 12

happy = [ "I am so thrilled and joyful today!", "This is wonderful, I feel fantastic!", "What a delightful, cheerful morning!", "I'm overjoyed, everything is amazing!", ] 
sad = [ "I am so miserable and hopeless today.", "This is awful, I feel terrible.", "What a bleak, depressing morning.", "I'm devastated, everything is ruined.", ]

def mean_act(sentences):
    acts = [activation(s, LAYER) for s in sentences]   # each is [896]
    return torch.stack(acts).mean(dim=0)               # average -> [896]


def activation(text, layer):
    ids = tok(text, return_tensors="pt").to(model.device)
    # hidden_states is a tuple: [embeddings, after-layer-1, after-layer-2, ...]
    hs = model(**ids, output_hidden_states=True).hidden_states
    return hs[layer][0, -1]   # the LAST token's vector at this layer

happy_dir = mean_act(happy) - mean_act(sad)   

# Hook target: the module whose output IS hidden_states[LAYER].
# hs[L] is the output of layers[L-1], so we hook layers[LAYER-1].
target_layer = model.model.layers[LAYER - 1]

def make_hook(vec):
    v = vec.to(model.dtype)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + v,) + out[1:]   # old style: (hidden_states, ...)
        return out + v                        # new style: bare tensor
    return hook

def generate(prompt, steer_vec=None, max_new=60):
    handle = target_layer.register_forward_hook(make_hook(steer_vec)) if steer_vec is not None else None
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        if handle:
            handle.remove()

PROMPT = "Tell me about going to the grocery store."

for mult in (0, 2, 4):
    vec = None if mult == 0 else happy_dir * mult
    print(f"\n===== happy strength x{mult} =====")
    print(generate(PROMPT, vec))