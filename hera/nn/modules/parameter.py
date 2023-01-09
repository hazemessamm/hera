
class Parameter:
    def __init__(self, rng, initializer, shape):
        self.data = None
        self.rng = rng
        self.initializer = initializer
        self.shape = shape
        self._name = None

    def reset_parameter(self):
        if self.shape is None:
            self.data = ()
        else:
            self.data = self.initializer(self.rng, self.shape)

    def update_parameter(self, new_data, trainable=True):
        if isinstance(new_data, tuple) and len(new_data) == 0:
            return
        if trainable:
            if new_data.shape == self.data.shape:
                self.data = new_data
            else:
                raise ValueError(
                    f"Shape mismatch. {new_data.shape} != {self.data.shape}"
                )
