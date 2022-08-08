# Week 2: Neural Network Training

**Overview**:

-   Train a neural network on data using TensorFlow
-   Understand the difference between various activation functions (sigmoid, ReLU, and linear)
-   Understand which activation functions to use for which type of layer
-   Understand why we need non-linear activation functions
-   Understand multiclass classification
-   Calculate the softmax activation for implementing multiclass classification
-   Use the categorical cross entropy loss function for multiclass classification
-   Use the recommended method for implementing multiclass classification in code
-   (Optional): Explain the difference between multi-label and multiclass classification

---

## Ch 1: Neural Network Training

### Tensorflow Implementation

So, now we know how to create a **Neural Network** in _Python_ using _Tensorflow_. Let's see how we can train a **neural network** using _Tensorflow_.

We'll use _Hand-written Digit Recognition_ example.

It's **neural network** architecture is:

<img src="./images/hand_digit_nn.jpg" alt="handwritten digit neural network" width="500px" style="padding:10px 50px">

-   Here, we have
-   -   total $3$ layers
-   -   In $1^{st}$ layer we have $25$ neurons
-   -   In $2^{nd}$ layer we have $15$ neurons
-   -   In $3^{rd}$ _output_ layer, we have $1$ neuron.

#### Step 1: Creating Neural Network

```python
from tensorflow.keras import Sequential, layers.Dense

model = Sequential([
    layers.Dense(25, activation='sigmoid'),
    layers.Dense(15, activation='sigmoid'),
    layers.Dense(10, activation='sigmoid')
])
```

-   We are creating it's **neural network**.

#### Step 2: Compiling Neural Network

```python
from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentropy())
```

-   We are compiling our **neural network**.
-   Compiling the model means to specify the loss function which are going to use, here we are using `BinaryCrossentropy` loss function.

#### Step 3: Training Neural Network

```python
model.fit(X, Y, epochs=100)
```

-   We are using `model.fit` to train our **neural network** model.
-   `epoch` is same as the number of iterations in _gradient descent_.

---
