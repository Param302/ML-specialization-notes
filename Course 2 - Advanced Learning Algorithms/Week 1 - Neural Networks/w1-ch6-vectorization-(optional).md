# Week 1: Neural Networks

## Ch 6: Vectorization (optional)

One of the reasons, we can use **neural networks** so efficiently is because of **vectorization**. They can be implemented very efficiently using _matrix multiplication_.

By using parallel computing hardware, including GPUs, we can do very large _matrix multiplication_.

### Matrix Multiplication code

Let's see how _matrix multiplication_ code looks like:

By using _Coffee Roasting_ example, we have

1. Input $\mathbf{x}$

```python
import numpy as np
X = np.array([200, 17])
```

2. Weights $\mathbf{w}$

```python
W = np.array([
    [ 1, -3,  5],
    [-2,  4,  6]
])
```

3. Bias $\mathbf{b}$

```python
b = np.array([-1, 1, 2])
```

4. And, our `dense` function is:

```python
def dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(X, w) + b[j]
        a_out[j] = g(z)
    return a_out
```

-   But using _vectorization_ and _matrix multiplication_ we do it much faster.

1. We will convert the input to a matrix $\mathbf{X}$

```python
X = np.array([[200, 17]])
```

2. Weight was already a matrix $\mathbf{W}$

```python
W = np.array([
    [ 1, -3,  5],
    [-2,  4,  6]
])
```

3. We will convert Bias to matrix also $\mathbf{B}$

```python
B = np.array([-1, 1, 2])
```

4. And, our `dense` function is:

```python
def dense(A_in, W, B, g):
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out
```

5. We can use `np.matmul` to do _matrix multiplication_, which also makes the code shorter.
6. We can also use `@` (matmul) operator to do _matrix multiplication_ as well as for _dot product_.

```python
Z = X @ Y    # matrix multiplication
z = x @ y    # dot product
```

7. It also results the output in a matrix.

---

