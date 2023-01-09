import jax

from hera.nn.modules.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.relu(inputs)


class LeakyReLU(Module):
    def __init__(self, negative_slope=1e-2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs):
        return jax.nn.leaky_relu(inputs, self.negative_slope)


class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.gelu(inputs)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.sigmoid(inputs)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.tanh(inputs)


class GLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.glu(inputs)


class ELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.elu(inputs)


class LogSigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.log_sigmoid(inputs)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return jax.nn.softmax(inputs, self.axis)


class LogSoftmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        return jax.nn.log_softmax(inputs, self.axis)


class Softplus(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return jax.nn.softplus(inputs, self.axis)
