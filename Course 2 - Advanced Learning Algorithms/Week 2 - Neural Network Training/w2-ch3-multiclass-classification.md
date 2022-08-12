# Week 2: Neural Network Training

## Ch 3: Multiclass Classification

### What is Multiclass Classification?

> _Multiclass classification_ refers to the **classification** problems where you can have more than just two possible output labels so not just $0$ or $1$. ~ _Andrew Ng_

In _Multiclass classification_ problem, we have to classify the _target_ variable $y$ into multiple classes (more than $2$), whereas in _Binary classification_, we classify the target variable $y$ into two classes ($0$ or $1$).

If the target variable $y$ has $n$ no. of discrete values (where $n\not ={2}$), then we can use _Multiclass classification_.

---

#### Hand-written digit recognition

Earlier, in _Hand-written digit recognition_, we are classifying the target variable $y$ into $2$ classes $0$ or $1$, whether the digit is a zero or a one, which is considered as _Binary classification_ problem.

<img src="./images/binary_classification_example.jpg" alt="binary classification hand written digit example" width="200px" style="padding:10px">

Now, we will classify the _Hand-written digit_ into $10$ classes $0$ to $9$, which is _Multiclass classification_ problem.

<img src="./images/multiclass_classification_example.jpg" alt="multiclass classification hand written digit example" width="800px" style="padding:10px">

---

#### Graph Illustration

#### Binary Classification

In _Binary classification_ with $2$ _input_ features, graph looks like this:

<img src="./images/binary_classification.jpg" alt="binary classification graph" width="400px" style="padding:10px">

Where we need to find the probability of the target variable $y$ being $1$ i.e. $P\left(y=1|\vec{x}\right)$.

#### Multiclass Classification

In **Multiclass classification** with $2$ _input_ features, graph looks like this:

<img src="./images/multiclass_classification.jpg" alt="multiclass classification graph" width="400px" style="padding:10px">

Where we need to find probability of target variable $y$ being $1$, $2$, $3$ or $4$, i.e. $P\left(y=1|\vec{x}\right)$, $P\left(y=2|\vec{x}\right)$, $P\left(y=3|\vec{x}\right)$ and $P\left(y=4|\vec{x}\right)$.

And in the above graph, _decision boundary_ will look like this:

<img src="./images/multiclass_classification_2.jpg" alt="multiclass classification decision boundary graph" width="400px" style="padding:10px">

---

### Softmax Regression

> The _softmax regression_ is a generalization of _logistic regression_, which extends it from _binary classification_ to _multiclass classification_. ~ _Andrew Ng_

To understand how _softmax regression_ is _generalized version_ of _logistic regression_, let's see how _logistic regression_ works.

---

#### Logsitc Regression Interpretation

In _Logistic Regression_, we classify the target variable $y$ into $2$ classes $0$ or $1$, which is by applying the threshold of $0.5$ to the output of _sigmoid_ function.

1. First, we have to compute the value of $z$.
   $$z = \vec{w} \cdot \vec{x} + b$$

2. And, then calculate the value of $sigmoid(z)$, which gives us the probability of $y$ being $1$ for the given _input_ features $x$.
   $$a = g(z) = \frac{1}{1 + e^{-z}} \qquad P\left(y=1 | \vec{x}\right)$$
3. We know that the sum of all the probabilities is equal to $1$.
4. So, we can also say that the probability of $y$ being $0$ is $1 - a$ for the given _input_ features $x$.
   $$a_2 = 1 - a_1 \qquad P\left(y=0 | \vec{x}\right)$$

---

#### Softmax Regression Interpretation

In _Softmax Regression_, we classify the target variable $y$ into $n$ classes.

Let's say $n=4$, means that we have $4$ classes, i.e. $y=1$, $y=2$, $y=3$ and $y=4$.

For each value of $y$, _softmax regression_ will compute the probability of $y$ being $1$, $2$, $3$ or $4$.

1. First, it will compute all the $4$ values of $z$ for the given _input_ features $x$, with different parameters $w$ and $b$ for each class.
   $$z_1 = \vec{w}_1 \cdot \vec{x} + b_1\tag1$$
   $$z_2 = \vec{w}_2 \cdot \vec{x} + b_2\tag2$$
   $$z_3 = \vec{w}_3 \cdot \vec{x} + b_3\tag3$$
   $$z_4 = \vec{w}_4 \cdot \vec{x} + b_4\tag4$$

2. For each value of $z$, it will compute the values of $a$ for the given _input_ features $x$.
   $$a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}\tag1$$
   $$a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}\tag2$$
   $$a_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}\tag3$$
   $$a_4 = \frac{e^{z_4}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}\tag4$$
3. Where, each $a$ is the probability of $y$ being $1$, $2$, $3$ or $4$ for the given _input_ features $x$.
   $$a_1 = P\left(y=1 | \vec{x}\right)\tag1$$
   $$a_2 = P\left(y=2 | \vec{x}\right)\tag2$$
   $$a_3 = P\left(y=3 | \vec{x}\right)\tag3$$
   $$a_4 = P\left(y=4 | \vec{x}\right)\tag4$$

---

### Softmax regression formula

So, we've seen how _softmax regression_ works for $4$ classes, i.e. $4$ values of the _target_ variable $y$.

Now, let's see the formula for _softmax regression_ for $N$ classes, i.e. $N$ values of the _target_ variable $y$.

1. For each value of $y$, _softmax regression_ will compute the value of $z$.
   $$z_j = \vec{w}_j \cdot \vec{x} + b_j$$
   $$where \quad j = 1 ... N$$

2. And, for each value of $z_j$, we have different values of parameters $w_j$ and $b_j$.

3. Finally, it computes the value of $a_j$.
   $$a_j = \frac{e^{z_j}}{\sum_{k=1}^{N}e^{z_k}} \quad= P\left(y=j|\vec{x}\right)$$

4. Where, $k$ is the index of each value of $e^{z_k}$ for the summation, because $j$ is fixed.

---

### Softmax Regression as a generalization of Logistic Regression

> if we apply _softmax regression_ with $n$ equals $2$, so there are only two possible output classes then _softmax regression_ ends up computing basically the same thing as _logistic regression_. The parameters end up being a little bit different, but it ends up reducing to _logistic regression_ model. But that's why the _softmax regression_ model is the generalization of _logistic regression_. ~ _Andrew Ng_

---

### Cost Function

#### Logistic Cost function

In _Logistic Regression_'s, the _loss function_ is:
$$Loss = - y \log(a_1) - (1 - y) \log(\overbrace{1 - a_1}^{a_2})$$

We know that $a_2 = 1 - a_1$, which is the probability of $y$ being $0$.

$$Loss = - y \log(a_1) - (1 - y) log(a_2)$$

Where, $-y\log(a_1)$ will result when $y=1$, and $-y\log(a_2)$ will result when $y=0$.
$$Loss = \underbrace{- y \log(a_1)}_{y=1} - (1 \underbrace{- y) log(a_2)}_{y=0}$$

The _loss_ will be calculated for each value of $y$ and it's _cost function_ will be calculated by calculating the average of all the loss values.

---

#### Softmax Cost function

The _loss function_ for _softmax regression_ is:
$$loss(a_1,...a_n) = \begin{cases} - \log a_1\quad\text{if } y = 1 \\ -\log a_2\quad\text{if }y = 2 \\ \qquad\qquad\vdots \\ - \log a_N\quad\text{if } y = N\end{cases}$$

In the _loss function_, $y$ can take up any value till $N$ which is denoted by $j$, so is $a_j$.

If $y=N$, then it's loss will be $-\log a_N$.

It's graph looks like this:

<img src="./images/softmax_loss_function_graph.jpg" alt="softmax loss function graph" width="400px" style="padding:10px">

-   If $a_j$ is very close to $j$, i.e. $a_1 \approx 1$, then the loss $L$ will be close to $0$ (very less).
-   If $a_j$ is $0.5$, then the loss $L$ will be in the mid.

So, smaller the value of $a_j$, the more the loss will be.

---

### Neural Network with _Softmax_ activation function

Let's take the _Hand-written Digit Recognition_ example.

<img src="./images/multiclass_classification_example.jpg" alt="multiclass classification hand written digit example" width="800px" style="padding:10px">

Here we have $10$ classes, i.e. $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$ for each digit.

So, we want a **neural network** which outputs the probability of each class being correct. And, it's _output_ layer must have $10$ neurons and _softmax_ function as an _activation_ function.

So, we need a **neural network** like this:

<img src="./images/hand_digit_nn_multiple.jpg" alt="hand digit neural network softmax" width="500px" style="padding:10px">

Here, we have:

-   $3$ layers; $2$ _hidden_ and $1$ output layer.
-   _Output_ layer has $10$ neurons.

In _output_ layer, each _neuron_ computes the value $z$
$$z^{[3]}_j = \vec{w}^{[3]}_j \cdot \vec{a}^{[2]} + b^{[3]}_1$$

And, find the value of $a_j$
$$a^{[3]}_j = \frac{e^{z^{[3]}_1}}{e^{z^{[3]}_1} + \cdots + e^{z^{[3]}_{10}}}$$

Here, one thing is different, i.e. in other _activation_ functions, we have a single _activation_ value $a_i$ for each _input_ value $x_i$, but in _softmax_ function, we have $a_j$ which contains the probability of each _input_ $x_i$ being the class $j$.

In _logistic regression_, we use a single $z$ value for computing an _activation_ value:
$$a^{[3]}_j = g\left(z^{[3]}_j\right)$$

But, in _softmax regression_, we use **all** $z$ values for computing a single _activation_ value:
$$\vec{a}^{[3]} = g\left(z^{[3]}_1, ..., z^{[3]}_N\right)$$
where $\vec{a}^{[3]}$ holds all the $a_j$ values upto $N$:
$$\vec{a}^{[3]} = \left[a^{[3]}_1, ..., a^{[3]}_N\right]$$

#### Example

We have _input_ features $x = [[200, 15], [250, 20]], in _binary classification_ or _regression_ problem, we have output like this:

**Classification**: $\hat{y} = [0, 1]$

**Regression**: $\hat{y} = [300, 170]$

But, in _multiple classification_ problem, we have _input_ features $x = [[240, ... 200], [250, ... 200]]$, so it's output will be like this:
$$\hat{y} = [[0.2, 0.5, 0.7, 0.3], [0.1, 0.8, 0.2, 0.5]]$$

where, each row is the probability of each class.

---

#### Tensorflow code with _Softmax_

We can make a **neural network** with _softmax_ activation function by using the following code:

1. Import required libraries

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
```

2. Create a **neural network**

```python
model = Sequential([
   Dense(25, activation="relu"),
   Dense(15, activation="relu"),
   Dense(10, activation="softmax")
])
```

3. Here, we are creating $3$ layers, consists of $2$ _hidden_ with **ReLU** as their _activation_ function and $1$ _output_ with _softmax_ _activation_ function.
4. Specifying the _loss function_

```python
model.compile(loss=SparseCategoricalCrossentropy())
```

5. Here, we are using `SparseCategoricalCrossentropy` as _loss function_.

    > #### `SparseCategoryCrossentropy`
    >
    > It is a _loss_ function to compute _loss_ for _multiple classes_.
    >
    > _Categorical_ means we are classifying the _output_ $\hat{y}$ into categories. In _Hand-written Digit Recognition_ example, $10$ differen classes.
    >
    > _Sparse_ means output $\hat{y}$ can take one of the $N$ values. In _Hand-written digit recognition_, the output can be one of the $0, 1, 2, 3, 4, 5, 6, 7, 8, 9$ values.

6. Train the model

```python
model.fit(X, Y, epochs=100)
```

---
