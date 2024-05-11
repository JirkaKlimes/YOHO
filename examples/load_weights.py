from safetensors import safe_open

weights = safe_open("./weights/model_base_multi.safetensors", framework="flax")

for w_name in weights.keys():
    shape = weights.get_tensor(w_name).shape
    print(f"{str(shape):<14} {w_name}")
