First, we have the `nn` directory which holds all neural network related classes
like the modules, losses, optimizers and training utilities.

In the `nn` directory you will find 3 directories
(the number of directories may change in the future)
which are `losses`, `optimizers` and `modules`.

1. All of the essential neural network modules exists in
the `module` directory like `Conv1D`, `Conv2D`, `Linear`, `Dropout`, etc...,
if there is multiple modules that does a certain functionality
it will be in a seperate directory inside the module directory
(e.g. Convolution operations like `Conv1D` and `Conv2D` exists in the `convolution` directory.)

2. All the essential neural network optimizers exists in the `optimizers` directory.
`base_optimizer.py` is the base optimizer class that all of our optimizers inheret from.

3. All the essential neural network losses exists in the `losses` directory.
`base_loss.py` is the base loss class for all of our losses inheret from.