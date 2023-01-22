import unittest
from jax import numpy as jnp
import jax
from hera.nn.modules.pooling.average_pooling_2d import AvgPooling2D
import hera

hera.backend.enable_auto_register(True)

class AveragePooling2DTest(unittest.TestCase):
    def test_inputs_output_shapes(self):
        inputs = jax.random.normal(jax.random.PRNGKey(3), (1, 20, 20, 3))
        avgpool2d = AvgPooling2D(2)
        out = avgpool2d(inputs)
        self.assertEqual(out.shape, (1, 10, 10, 3))

    def test_inputs_output_equal(self):
        inputs = jax.random.normal(jax.random.PRNGKey(3), (20, 150, 150, 3))
        avgpool2d = AvgPooling2D(2)
        out = avgpool2d(inputs)
        self.assertEqual(jnp.mean(inputs[0][:2, :2, 0]), out[0][0][0][0])
    
    def test_init(self):
        with self.assertRaises(ValueError):
            AvgPooling2D(0)
        
        with self.assertRaises(ValueError):
            AvgPooling2D(pool_size=())
        
        with self.assertRaises(ValueError):
            AvgPooling2D(pool_size=(0,))
        
        with self.assertRaises(ValueError):
            AvgPooling2D(pool_size=(1,), strides=(0,))

        with self.assertRaises(ValueError):
            AvgPooling2D(pool_size=(1,), strides=(1,), padding='not_valid')
        
        # with self.assertRaises(ValueError):
            # AvgPooling2D(pool_size=(1,), strides=(1,), padding=(0,))
        



if __name__ == '__main__':
    unittest.main()
