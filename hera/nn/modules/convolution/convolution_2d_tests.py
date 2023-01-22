import unittest
from jax import numpy as jnp
import jax
from hera.nn.modules.convolution.convolution_2d import Conv2D
import hera

hera.backend.enable_auto_register(True)

class AveragePooling2DTest(unittest.TestCase):
    def test_inputs_output_shapes(self):
        inputs = jax.random.normal(jax.random.PRNGKey(3), (1, 20, 20, 4))
        conv2d = Conv2D(4, 3, 3, 3)
        out = conv2d(inputs)
        self.assertEqual(out.shape, (1, 18, 18, 3))

    def test_inputs_output_same_padding(self):
        inputs = jax.random.normal(jax.random.PRNGKey(3), (20, 150, 150, 3))
        conv2d = Conv2D(3, 3, 3, 3, padding='same')
        out = conv2d(inputs)
        self.assertEqual(out.shape, (20, 150, 150, 3))
    
    def test_init(self):        
        with self.assertRaises(ValueError):
            Conv2D(1, 1, 1, 1, strides=(1,1), padding='not_valid')
        



if __name__ == '__main__':
    unittest.main()
