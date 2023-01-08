# Hera
Deep Learning library bulit on top of JAX and inspired from PyTorch



# Example 1:
```python
from hera import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__(jit=True)
        self.embedding = nn.Embedding(100, 128, 4)
        self.dense_1 = nn.Linear(128, 128, 5)
        self.dropout = nn.Dropout(0.7, 2)
        self.dense_2 = nn.Linear(128, 128, 6)
        self.mha = nn.MultiHeadAttention(128, 2, 5, 0.2)
        self.layernorm = nn.LayerNormalization(128, 10, True, True)
        self.transformer_encoder = nn.TransformerEncoderLayer(128, 4, 5)
        self._seq = Sequential(
            [nn.Linear(128, 128, 7), nn.GELU(), nn.GlobalAvgPooling1D(), nn.Linear(128, 1, 8)])

    def forward(self, weights, x):
        out = self.embedding(weights["embedding"], x)
        out = self.dense_1(weights["dense_1"], out)
        out = jax.nn.gelu(out)
        out = self.dropout(out)
        out = self.dense_2(weights["dense_2"], out)
        out = jax.nn.gelu(out)
        out = self.mha(weights["mha"], out, out, out)
        out = self.layernorm(weights["layernorm"], out)
        out = self.transformer_encoder(weights["transformer_encoder"], out)
        out = self._seq(weights["_seq"], out)
        return out