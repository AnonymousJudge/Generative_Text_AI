import jax
import jax.numpy as jnp
import jax.random as random
import json
from pathlib import Path

def serialize_params(params):
    return [w.tolist() for w in params]

def deserialize_params(data):
    return [jnp.array(w) for w in data]

class NeuralNetworkJAX:
    def __init__(self, layers, alpha=0.1, seed=0, path=None):
        self.layers = layers
        self.alpha = alpha
        self.key = random.PRNGKey(seed)
        self.path = path

        if path and Path(path).exists():
            self.params = self.load(path)
        else:
            self.params = self.init_weights()

    def init_weights(self):
        keys = random.split(self.key, len(self.layers) - 1)
        weights = []
        for i, k in enumerate(keys):
            fan_in = self.layers[i] + 1
            fan_out = self.layers[i + 1] if i == len(self.layers) - 2 else self.layers[i + 1] + 1
            W = random.normal(k, (fan_in, fan_out)) / jnp.sqrt(fan_in)
            weights.append(W)
        return weights

    def save(self, path=None):
        save_path = path or self.path
        if not save_path:
            raise ValueError("No path provided to save weights.")
        with open(save_path, "w") as f:
            json.dump(serialize_params(self.params), f)

    def load(self, path):
        load_path = path or self.path
        if not load_path:
            raise ValueError("No path provided to load weights.")
        with open(path, "r") as f:
            data = json.load(f)
        return deserialize_params(data)

    def predict(self, X):
        return feedforward(self.params, X)

    def train(self, X, y, epochs=1000, displayUpdate=100):
        self.params = train(self.params, X, y, epochs, self.alpha, displayUpdate)

    def layer_vector(self, x, layer_index = 0):
        return get_layer_output(self.params, x, layer_index)

def init_weights(key, layers):
    keys = random.split(key, len(layers) - 1)
    weights = []

    for i, k in enumerate(keys):
        fan_in = layers[i] + 1
        fan_out = layers[i + 1] if i == len(layers) - 2 else layers[i + 1] + 1
        W = random.normal(k, (fan_in, fan_out)) / jnp.sqrt(fan_in)
        weights.append(W)
    
    return weights

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def sigmoid_deriv(output):
    return output * (1 - output)

def feedforward(params, X):
    if X.ndim == 1:
        X = X[None, :]  # make it 2D if single example

    a = X
    for i, W in enumerate(params):
        # Append bias if weight expects it
        if a.shape[1] + 1 == W.shape[0]:
            a = jnp.concatenate([a, jnp.ones((a.shape[0], 1))], axis=1)
        a = jnp.dot(a, W)
        a = sigmoid(a)
    return a

def loss_fn(params, X, y):
    preds = feedforward(params, X)
    return 0.5 * jnp.mean(jnp.square(preds - y))

@jax.jit
def train_step(params, X, y, alpha):
    grads = jax.grad(loss_fn)(params, X, y)
    new_params = [W - alpha * dW for W, dW in zip(params, grads)]
    return new_params

def train(params, X, y, epochs=1000, alpha=0.1, displayUpdate=100):
    for epoch in range(epochs):
        params = train_step(params, X, y, alpha)
        if epoch == 0 or (epoch + 1) % displayUpdate == 0:
            l = loss_fn(params, X, y)
            print(f"[INFO] epoch={epoch+1}, loss={l:.10f}")
    return params

def get_layer_output(params, x, layer_index):
    """
    Forward input `x` up to the given `layer_index` and return the activation.

    Parameters:
        params: List of weight matrices
        x: Input array of shape (n_features,) or (batch_size, n_features)
        layer_index: Index of the layer to stop at (0 = first hidden)

    Returns:
        JAX array of activations at the specified layer
    """
    if x.ndim == 1:
        x = x[None, :]  # convert to batch of size 1
    a = x
    a = jnp.concatenate([a, jnp.ones((a.shape[0], 1))], axis=1)  # bias input

    for i, W in enumerate(params):
        a = jnp.dot(a, W)
        a = sigmoid(a)
        if i == layer_index:
            return a
        if i < len(params) - 1:
            a = jnp.concatenate([a, jnp.ones((a.shape[0], 1))], axis=1)

    raise ValueError("layer_index out of bounds")