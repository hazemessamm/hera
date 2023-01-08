# Hera
Deep Learning library bulit on top of JAX and inspired from PyTorch



# Example 1:
```python
from hera import nn

# Implement the model like PyTorch.
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__(jit=True)
        self.conv_1 = nn.Conv2D(1, 32, 3, 3, activation=jax.nn.relu)
        self.dropout_1 = nn.Dropout(0.2, 5)
        
        self.conv_2 = nn.Conv2D(32, 32, 3, 4, activation=jax.nn.relu)
        self.dropout_2 = nn.Dropout(0.2, 5)

        self.conv_3 = nn.Conv2D(32, 32, 3, 4, activation=jax.nn.relu)
        self.dropout_3 = nn.Dropout(0.2, 5)

        self.flatten = nn.Flatten()

        output_shape = self.flatten.compute_output_shape(
            self.conv_3.compute_output_shape(
                self.conv_2.compute_output_shape(
                    self.conv_1.compute_output_shape((1, 28, 28, 1))
                )
            )
        )

        self.dense_1 = nn.Linear(output_shape[-1], 128, 6)
        self.dense_2 = nn.Linear(128, 10, 7)

    def forward(self, weights, x):
        # When we create our nested layers
        # we also create a dictionary with
        # the attribute names (conv_1, dropout_1, etc..)
        # as keys and their weights as values
        # then we pass each weight to its corresponding layer
        # to allow JAX to track them for backward propagation.
        out = self.conv_1(weights["conv_1"], x)
        out = self.dropout_1(weights["dropout_1"], out)
        out = self.conv_2(weights["conv_2"], out)
        out = self.dropout_2(weights["dropout_2"], out)
        out = self.conv_3(weights["conv_3"], out)
        out = self.dropout_3(weights["dropout_3"], out)
        out = self.flatten(weights["flatten"], out)
        out = self.dense_1(weights["dense_1"], out)
        out = self.dense_2(weights["dense_2"], out)
        return out

    
    model = MnistModel()

    # Load the dataset and prepare it.
    train_df = pd.read_csv('mnist_train.csv')
    labels = train_df.iloc[:, 0].values
    train_data = np.reshape(train_df.iloc[:, 1:].values,
                            (-1, 28, 28, 1)).astype('float32')

    test_df = pd.read_csv('mnist_test.csv')
    test_data = np.reshape(test_df.values, (-1, 28, 28, 1)).astype('float32')


    # Backward function that returns the loss, predictions and the gradients.
    @partial(jax.value_and_grad, has_aux=True)
    def backward(weights, x, y, loss_fn):
        out = model(weights, x)
        loss = loss_fn(out, y)
        return loss, out


    # Our training function
    def train(model: nn.Module, steps, batch_size, train_data,
              train_labels, test_data):

        # Our labels are integers so this loss function will work with it.
        loss_fn = nn.SparseCrossEntropyLoss()
        
        # Adam Optimizer.
        optimizer = hera.optimizers.Adam(0.001)
        
        with tqdm(range(steps), leave=True) as t:
            for step in t:
                # Sample random examples
                ids = np.random.randint(0, train_data.shape[0], (batch_size,))
                batch_data = train_data[ids, :]
                batch_labels = train_labels[ids]
                 
                params = model.parameters()

                # Apply backward propagation.
                (loss, preds), grads = backward(model.parameters(),
                                                batch_data, batch_labels, loss_fn)
                # update the weights
                new_weights = optimizer.update_weights(grads, params)
                
                # Pass the new weights to the model.
                model.update_parameters(new_weights=new_weights)
                    t.set_description(f'Loss: {round(loss, 4)}')

        # Instead of model.eval() and model.train() (They are available.) 
        # we use `eval_mode` context manager which lets the model enters 
        # evaluation mode then automatically
        # returns to training mode after exiting from it.
        with hera.eval_mode(model):
            ids = np.random.randint(0, test_data.shape[0], (batch_size,))
            batch_data = train_data[ids, :]
            out = model(model.parameters(), batch_data).argmax(-1)
```
