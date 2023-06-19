# Hera
Deep Learning library bulit on top of JAX and inspired from PyTorch



### Example 1:
```python
import hera
from hera import nn

hera.set_global_rng(5)

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__(jit=True)

        self.convs = nn.Sequential([nn.Conv2D(1, 32, 3, activation=jax.nn.gelu),
                                    nn.Dropout(0.2),
                                    nn.Conv2D(32, 32, 3, activation=jax.nn.gelu),
                                    nn.Dropout(0.2),
                                    nn.Conv2D(32, 32, 3, activation=jax.nn.gelu),
                                    nn.Dropout(0.2),])

        self.flatten = nn.Flatten()
        output_shape = self.flatten.compute_output_shape(
            self.convs.compute_output_shape((1, 28, 28, 1)))

        self.dense_1 = nn.Linear(output_shape[-1], 128, activation=jax.nn.gelu)
        self.dense_2 = nn.Linear(128, 10)

    def forward(self, weights, x):
        out = self.convs(weights['convs'], x)
        out = self.flatten(weights['flatten'], out)
        out = self.dense_1(weights['dense_1'], out)
        out = self.dense_2(weights['dense_2'], out)
        return out


model = MnistModel()

# Load the dataset and prepare it.
train_df = pd.read_csv('mnist_train.csv')
labels = train_df.iloc[:, 0].values
train_data = np.reshape(train_df.iloc[:, 1:].values,
                        (-1, 28, 28, 1)).astype('float32')

test_df = pd.read_csv('mnist_test.csv')
test_data = np.reshape(test_df.values, (-1, 28, 28, 1)).astype('float32')


# Our training function
def train(model: nn.Module, steps, batch_size, train_data,
            train_labels, test_data):

    # Our labels are integers so this loss function will work with it.
    loss_fn = nn.SparseCrossEntropyLoss()
    
    # Adam Optimizer.
    optimizer = hera.optimizers.Adam(model, 0.001)
    
    with tqdm(range(steps), leave=True) as t:
        for step in t:
            # Sample random examples
            ids = np.random.randint(0, train_data.shape[0], (batch_size,))
            batch_data = train_data[ids, :]
            batch_labels = train_labels[ids]

            # Apply backward propagation.
            with hera.BackwardRecorder(model=model, loss=loss_fn, optimizer=optimizer, auto_zero_grad=True) as recorder:
                loss_val, predictions = recorder(batch_data, targets=batch_labels)
                recorder.step()

            t.set_description(f'Loss: {round(loss, 4)}')


    # Instead of model.eval()
    # and model.train() (They are available.) 
    # we use `eval_mode` context manager
    # which lets the model enters 
    # evaluation mode then automatically
    # returns to training mode after
    # exiting from the context manager.
    with hera.eval_mode(model):
        ids = np.random.randint(0, test_data.shape[0], (batch_size,))
        batch_data = train_data[ids, :]
        out = model(model.parameters(), batch_data).argmax(-1)
```

### Sequential Model Example:
```python
model = nn.Sequential([
    nn.Conv2D(1, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Conv2D(32, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Conv2D(32, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Flatten(),
    nn.Linear(15488, 128),
    nn.Linear(128)
])

# But what if you want to calculate the output shape after flatten module?
# You can do the following:

sequential_model = nn.Sequential([
    nn.Conv2D(1, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Conv2D(32, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Conv2D(32, 32, 3, activation=jax.nn.relu),
    nn.Dropout(0.2),
    nn.Flatten()
])

# Calcuate the output shape
out_shape = sequential_model.compute_output_shape((1, 28, 28, 1))

# Then add the last two layers using the `.add()` method.
sequential_model.add(nn.Linear(out_shape[-1], 128, 9, activation=jax.nn.relu))
sequential_model.add(nn.Linear(128, 10, 10))

# You can also do the same using:
sequential_model.add_modules([nn.Linear(out_shape[-1], 128, 9, activation=jax.nn.relu), nn.Linear(128, 10, 10)])

```
