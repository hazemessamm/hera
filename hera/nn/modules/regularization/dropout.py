from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Dropout(Module):
    def __init__(self, rate: float, rng: int):
        """Dropout Module

        Args:
            rate (float): Dropout probability between zero and one.
            rng (int): Initial seed that will be used to create another
                       random seeds each dropout call.
        """

        # Stochastic module is set to True only in the case of
        # requiring different random number every time we call it.
        super().__init__(rng, non_deterministic=True)
        self.rate = rate

    def pre_forward_hook(self, *args, **kwargs):
        random_key = self.make_random_key()
        return (random_key,)

    def forward(self, inputs, rng):
        out = F.dropout(inputs, self.rate, rng, self.training)
        return out
