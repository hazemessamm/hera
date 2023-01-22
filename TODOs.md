### Modules to be added.

- [ ] Average Pooling
- [ ] Transformer Decoder Layer
- [ ] Transformer Encoder
- [ ] Transformer Decoder
- [ ] Batch Normalization
- [ ] Group Normalization
- [ ] More activation functions
- [ ] Loss functions in functional module
- [ ] NLLLoss
- [ ] BCELoss
- [ ] CTCLoss
- [ ] Locally Connected modules
- [ ] Multihead Attention with RoPE module
- [ ] Multihead Attention with relative positional embedding module
- [ ] RNN module
- [ ] LSTM module
- [ ] GRU module
- [ ] Hinge Loss
- [ ] Huber Loss
- [ ] KL Divergence
- [ ] Cosine Similarity
- [ ] Poisson Loss
- [ ] RMSprop Optimizer
- [ ] SGD Optimizer
- [ ] Adagrad Optimizer
- [ ] Module Dict
- [ ] ConvBert
- [ ] Serialization
- [ ] Saving & Loading weights
- [ ] Convolution Transpose

You are welcome to contribute by adding any of these missing modules or
if there is another module that you want to add you can submit a pull request or an issue.

Note: Before implementing any of the mentioned loss functions, 
check [optax](https://github.com/deepmind/optax),
It's recommended to use it and you might find the loss function implemented
and you will just need to wrap it with the `hera.losses.Loss` class.


Note: Before implementing any of the mentioned optimizers, 
check [optax](https://github.com/deepmind/optax), 
It's recommended to use it and you might find the optimizer implemented
and you will just need to wrap it with the `hera.optimizers.Optimizer` class.

