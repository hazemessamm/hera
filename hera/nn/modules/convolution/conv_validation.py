

def _canonicalize_tuple(x, repeat=1):
    return (x,) * repeat

def validate_conv1d_init(self):
    if isinstance(self.kernel_size, tuple):
        if len(self.kernel_size) == 0 or len(self.kernel_size) > 1:
            msg = "`kernel_size` should be a tuple with one element or an integer"
            raise ValueError(msg)
    elif isinstance(self.kernel_size, int):
        self.kernel_size = _canonicalize_tuple(self.kernel_size)

    if isinstance(self.strides, tuple):
        if len(self.strides) == 0 or len(self.strides) > 1:
            raise ValueError(
                "`strides` should be a tuple with one element or an integer"
            )
    elif isinstance(self.strides, int):
        self.strides = _canonicalize_tuple(self.strides)

    if self.padding == "causal":
        raise ValueError("Causal padding is currently not supported yet.")



def validate_conv2d_init(self):
    if isinstance(self.kernel_size, tuple):
        if len(self.kernel_size) == 0 or len(self.kernel_size) > 2:
            msg = "`kernel_size` should be a tuple with two elements or an integer"
            raise ValueError(msg)

        elif len(self.kernel_size) == 1:
            kernel_size = self.kernel_size[0]
            self.kernel_size = _canonicalize_tuple(kernel_size, repeat=2)
    
    elif isinstance(self.kernel_size, int):
        self.kernel_size = _canonicalize_tuple(self.kernel_size, repeat=2)
        
    if isinstance(self.strides, tuple):
        if len(self.strides) == 0 or len(self.strides) > 2:
            msg = "`strides` should be a tuple with two elements or an integer"
            raise ValueError(msg)

        elif len(self.strides) == 1:
            strides = self.strides[0]
            self.strides = _canonicalize_tuple(strides, repeat=2)
    
    elif isinstance(self.strides, int):
        self.strides = _canonicalize_tuple(self.strides, repeat=2)

    if self.padding == "causal":
        raise ValueError("Causal padding is currently not implemented in Conv2D.")
    
    
    

