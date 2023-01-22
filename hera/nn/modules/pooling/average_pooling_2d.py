from jax.numpy import ndarray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from jax import lax

class AvgPooling2D(Module):
    def __init__(self, pool_size, strides=None, padding="valid"):
        """Average Pooling Module."""
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        if not self._reconstructed_from_unflatten:
            self.validate_init()

    
    def validate_init(self):
        if isinstance(self.pool_size, int):
            if self.pool_size <= 0:
                raise ValueError(f"`pool_size` should be a integer with value bigger than zero or a tuple with two elements bigger than zero. Recieved {self.pool_size}")
            self.pool_size = (self.pool_size, self.pool_size)
        elif isinstance(self.pool_size, tuple):
            if 1 <= len(self.pool_size) < 2:
                self.pool_size += self.pool_size
            elif len(self.pool_size) > 2 or len(self.pool_size) < 1:
                raise ValueError(f'`pool_size` should be a tuple with length of 2. Recieved {self.pool_size}')
        else:
            raise ValueError(f'Expected `pool_size` to be a tuple with length of 2 or an integer. Recieved {self.pool_size}')

        if self.strides is None:
            self.strides = self.pool_size

        elif isinstance(self.strides, int):
            if self.strides <= 0:
                raise ValueError(f"`strides` should be a tuple with 2 values bigger than zero. Recieved {self.strides}")
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple):
            if 1 <= len(self.strides) < 2:
                self.strides += self.strides
            elif len(self.strides) > 2 or len(self.strides) < 1:
                raise ValueError(f'`strides` should be a tuple with length of 2. Recieved {self.strides}')
        else:
            raise ValueError(f'Expected `strides` to be a tuple with length of 2 or an integer. Recieved {self.strides}')

        if any(ps <= 0 for ps in self.pool_size):
            raise ValueError(f'`pool_size` should be a tuple of values where each value should be bigger than or equal to 1. Recieved {self.pool_size}')

        if any(s <= 0 for s in self.strides):
            raise ValueError(f'`strides` should be a tuple of values where each value should be bigger than or equal to 1. Recieved {self.strides}')
        
        if isinstance(self.padding, str):
            if self.padding.lower() not in {'valid', 'same'}:
                raise ValueError('`padding` should be a string with values'
                                 f'`valid` or `same` or a tuple with length of 2. '
                                 f'Recieved {self.padding}')
            else:
                self.padding = self.padding.upper()
        elif isinstance(self.padding, (list, tuple)):
            if not any(isinstance(p, tuple) for p in self.padding):
                raise ValueError(f'`padding` should be a list of 4 tuples where each tuple should contain two elements. Recieved {self.padding}')
        else:
            raise ValueError(f'Expected `padding` to be a `str`, `list` or `tuple` of 4 `tuples` where each `t`uple should contain two elements. Recieved {self.padding}')


    def forward(self, inputs: ndarray):
        """Applies average pooling for spatial data.

        Args:
            inputs (ndarray): A 4D tensor with
                              axis order: (batch_size, height, width, channels)

        Returns:
            ndarray: A 4D tensor with axis order: (batch_size, height, width, channels)
        """
        if isinstance(self.padding, tuple):
            padding = self.padding
        elif isinstance(self.padding, str):
            padding = tuple(lax.padtype_to_pads(inputs.shape, self.pool_size, self.strides, self.padding))
            self.padding = padding
        else:
            padding = tuple(self.padding)
            self.padding = padding
        
        return F.average_pooling_2d(inputs, pool_size=self.pool_size, strides=self.strides, padding=padding)
