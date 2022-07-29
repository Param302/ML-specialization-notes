# Week 3 - Classification

## Ch 3 - Gradient Descent for Logistic Regression

### Gradient Descent

We use Gradient descent to find best values of parameters for our _logistic regression_ model which minimizes our cost function $J_{\vec{w},b}$.

-   As we know, our _cost function_ is:
    $$J_{\vec{w},b} = -\frac{1}{m}\sum^m_{i=1}\left[y^{(i)}\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1-y^{(i)})\log\left(1-f_{\vec{w},b}(\vec{x}^{(i)})\right)\right]$$

-   In _linear regression_, our _gradient descent_ algorithm is:

$$
\text{repeat until convergence }\{\\
w_j = w_j - \alpha\frac{\partial }{\partial w_j}J(\vec{w},b) \\
b = b - \alpha\frac{\partial }{\partial b}J(\vec{w},b)\\
\}
$$

1. $\frac{\partial }{\partial w_j}J(\vec{w},b)$ is:
   $$\frac{\partial }{\partial w_j}J(\vec{w},b) = \frac{1}{m}\sum^m_{i=1}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)x^{(i)}_j$$

2. $\frac{\partial }{\partial b}J(\vec{w},b)$ is:
   $$\frac{\partial }{\partial b}J(\vec{w},b) = \frac{1}{m}\sum^m_{i=1}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)$$

where, $j$ is each feature from $1$ to $n$.

If we put the values of $\frac{\partial }{\partial w_j}J(\vec{w},b)$ and $\frac{\partial }{\partial b}J(\vec{w},b)$ in _Gradient descent_ algorithm, we get:

> $$
> \text{repeat until convergence }\{\\
> w_j = w_j - \alpha\left[\frac{1}{m}\sum^m_{i=1}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)x^{(i)}_j\right] \\
> b = b - \alpha\left[\frac{1}{m}\sum^m_{i=1}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)\right]\\
> \}
> $$

This is the **gradient descent** algorithm for **logistic regression**.

You might be wondering this _gradient descent_ algorithm for _logistic regression_ is same as _gradient descent_ algorithm for _linear regression_.

The formula is same, but the function $f$ is different.

-   **Linear Regression** function $f$:
    $$f_{\vec{w}, b}(\vec{x}^{(i)}) = \vec{w}\cdot\vec{x} + b$$

-   **Logistic Regression** function $f$:
    $$f_{\vec{w}, b}(\vec{x}^{(i)}) = \frac{1}{1 + e^{-z}}$$

where $z$ is:
$$z = \vec{w}\cdot\vec{x} + b$$

And, the concepts we use in _gradient descent_ for _linear regression_ is same in _logistic regression_:

-   We montior the _learning rate_ of _gradient descent_ to find best learning rate.
-   We monitor the _cost function_ where it converges.
-   We use _Vectorization_ to make the code more efficient.
-   We use _Feature Scaling_ to make the variables range equally, which helps in model run fast.

---

### Jupyter lab: Gradient Descent for logistic regression [optional] [🔗](../codes/W3%20-%20L6%20-%20Gradient%20Descent%20for%20logistic%20regression.ipynb)

---

### Jupyter lab: Logistic regression with scikit-learn [optional] [🔗](../codes/W3%20-%20L7%20-%20Logistic%20regression%20with%20scikit-learn.ipynb)

---

### Quizzes

#### Practice quiz

##### Question 1

<img src="../quizzes/Quiz%20-%208%20Gradient%20descent%20for%20logistic%20regression.jpg" alt="practice quiz question 1" width="60%">
<details>
<summary>    
    <font size='3' color='#00FF00'>Answer to <b>question 1</b></font>
</summary>
<p>If you have selected option <em>a (The update steps look like the update steps for linear regression, but the definition of function <b>f</b> is different.)</em> then you are right!<br/><b>Explanation:</b><br/>For logistic regression, function <b>f</b> is the sigmoid function instead of a straight line.</p>
</details>
