from pathlib import Path
import re
from safetensors import safe_open
import jax.numpy as jnp


def parse_weights(path: Path):
    weights = safe_open(path, framework="flax")

    def parse_name(name):
        name = name.replace("model.", "")
        name = name.replace("decoder.layers.", "decoder.DecoderBlock_")
        name = name.replace("encoder.layers.", "encoder.EncoderBlock_")
        name = name.replace("encoder", "audio_encoder")
        name = name.replace("decoder", "text_decoder")
        name = name.replace("self_attn_layer_norm", "LayerNorm_0")
        name = name.replace("audio_encoder_attn_layer_norm", "LayerNorm_1")
        if "encoder" in name:
            name = name.replace("final_layer_norm", "LayerNorm_1")
        else:
            name = name.replace("final_layer_norm", "LayerNorm_2")

        name = name.replace("self_attn", "MultiHeadAttention_0")
        name = name.replace("audio_encoder_attn", "MultiHeadAttention_1")
        name = name.replace("k_proj", "key_proj")
        name = name.replace("q_proj", "query_proj")
        name = name.replace("v_proj", "value_proj")
        name = name.replace("embed_tokens.weight", "Embed_0.embedding")
        name = name.replace("embed_positions.weight", "positional_embedding")
        name = name.replace("layer_norm", "LayerNorm_0")
        name = re.sub(r"fc(\d+)", lambda m: f"Dense_{int(m.group(1))-1}", name)
        name = re.sub(r"conv(\d+)", lambda m: f"Conv_{int(m.group(1))-1}", name)
        if "orm" in name:
            name = name.replace("weight", "scale")
        else:
            name = name.replace("weight", "kernel")

        if "Dense" in name and "kernel" in name:
            permute = (1, 0)
        elif "Conv" in name and "kernel" in name:
            permute = (2, 1, 0)
        else:
            permute = None

        return tuple(name.split(".")), permute

    parsed = {}

    for name in weights.keys():
        tensor = weights.get_tensor(name)

        if name == "model.encoder.embed_positions.weight":
            continue

        name, permutation = parse_name(name)

        if permutation:
            tensor = jnp.permute_dims(tensor, permutation)

        parsed[name] = tensor

    return parsed
