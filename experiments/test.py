import jax.numpy as jnp

arr = jnp.arange(8 * 4).reshape((1, 8, 4))
print(arr)
# rep = jnp.tile(arr, 2)
rep = jnp.repeat(arr, 2, axis=-1)
print(rep)
