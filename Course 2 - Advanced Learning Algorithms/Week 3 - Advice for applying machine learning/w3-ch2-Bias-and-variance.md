# Week 3: Advice for Applying Machine Learning

## Ch 2: Bias and Variance

Let's recall what _bias_ and _variance_ are:

### Bias

> The inability for a machine learning model to capture the true relationship is called _bias_. ~ [_StatQuest with Josh Stammer_](https://youtu.be/EuBBz3bI-aA?t=136)

Means, if the model doesn't even fit the training data well, and on new data, it can't make good predictions.

We can also say that the model which _overfits_ the training data has _low bias_ and the model which _underfits_ the training data has _high bias_.

---

### Variance

> In Machine learning, the difference in fits between datasets in called _Variance_. ~ [_StatQuest with Josh Stammer_](https://youtu.be/EuBBz3bI-aA?t=245)

Means, if the model fits the training data well, but on new data it makes worse predictions, then, the difference is called _Variance_.

We can also say that the model which _overfits_ the traning data has **high variance**, and the model which _underfits_ the training data has **low variance.**

> In Machine learning, the ideal algorithm has _low bias_ and can accurately make the true relationship and it has _low variance_ by producing good predictions on new data. Means, _Generalized model_ is better than _Underfitted model_ and _Overfitted model_ is better than _Underfitted model_. ~ [_StatQuest with Josh Stammer_](https://youtu.be/EuBBz3bI-aA?t=322)

---

### Diagonising Bias and Variance with Regression line

#### High Bias

Let's say we have a normal _Linear Regression_ model with single _degree_ of _polynomial_, i.e.:
$$f_{\vec{w}, b}(x) = w_1x + b$$

And, after plotting the _regression line_ on the _training data_, we get the following plot:

<img src="./images/linear_regression_g1.jpg" alt="linear regression degree 1" height="300px" style="padding:10px">

Here, we can see the _regression line_ doesn't fit properly to the _training data_, so it's _cost_ $J_{train}(\vec{w}, b)$ is high, and it **underfits** the data.

And, if we plot _validation data_ on the same plot, we get the following plot:

<img src="./images/linear_regression_g4.jpg" alt="linear regression degree 1 validation" height="300px" style="padding:10px">

Here, we can see that the _regression line_ doesn't fit properly to the _validation data_ as well, so it's _cost_ $J_{cv}(\vec{w}, b)$ is high too.

So, this is a case of **High Bias**.

-   A characterstic of an algorithm with _high bias_ is, it underfits the _training data_ and it's _training cost_ is high as well as _validation cost_ is high.

---

#### High Variance

If we have a _Linear Regression_ model with _degree_ of _polynomial_ as $4$ i.e.:
$$f_{\vec{w}, b}(x) = w_1x + w_2x^2 + w_3x^3 + w_4x^4 + b$$

And, after plotting the _regression line_ on the _training data_, we get the following plot:

<img src="./images/linear_regression_g3.jpg" alt="linear regression degree 4" height="300px" style="padding:10px">

Here, we can see the _regression line_ fits really well to the _training data_, so it's _cost_ $J_{train}(\vec{w}, b)$ is low.

And, if we plot _validation data_ on the same plot, we get the following plot:

<img src="./images/linear_regression_g6.jpg" alt="linear regression degree 4 validation" height="300px" style="padding:10px">

Here, we can see that the _regression line_ doesn't fit to the _validation data_, so it's _cost_ $J_{cv}(\vec{w}, b)$ is very high and hence it **overfits** the data.

So, this is a case of **High Variance**.

-   A characterstic of an algorithm with _high variance_ is, the _validation cost_ is much higher than the _training cost_.

---

#### Right fit (Low Bias and Low Variance)

If we have a _Linear Regression_ model with _degree_ of _polynomial_ as $2$, i.e.:
$$f_{\vec{w}, b}(x) = w_1x + w_2x^2 + b$$

And, after plotting the _regression line_ on the _training data_, we get the following plot:

<img src="./images/linear_regression_g2.jpg" alt="linear regression degree 2" height="300px" style="padding:10px">

Here, we can see the _regression line_ fits well to the _training data_, so it's _cost_ $J_{train}(\vec{w}, b)$ is low.

And, if we plot _validation data_ on the same plot, we get the following plot:

<img src="./images/linear_regression_g5.jpg" alt="linear regression degree 2 validation" height="300px" style="padding:10px">

Here, we can see that the _regression line_ fits well to the _validation data_ as well and it's _cost_ $J_{cv}(\vec{w}, b)$ is less low but not much high.

So, this is a case of **Right fit**.

-   If the _training cost_ and _validation cost_ are both low, then the model fits well to the data and it's a **Right fit**, means it has _Low Bias_ and _Low Variance_.

---

### Diagonising Bias and Variance with Learning Curve

If we plot the _training cost_ and _validating cost_, having _degree_ of _polynomial_ on $x$-axis and _cost_ on $y$-axis, we get the following plot:

<img src="./images/polynomial_and_cost.jpg" alt="linear regression learning curve" height="300px" style="padding:10px">

Here, we can see that:

1. If we have very low _degree_ of _polynomial_, let's say $1$, then the _training cost_ and _validation cost_ is high, so it's a case of **High Bias**.

2. If we have very high _degree_ of _polynomial_, let's say $4$, then the _training cost_ is low but _validation cost_ is high, so it's a case of **High Variance**.

3. If we have _degree_ of _polynomial_ as $2$, then the _training cost_ and _validation cost_ is low, so it's a case of **Right fit**.

---

### Indicators of Bias and Variance

<img src="./images/polynomial_and_cost_2.jpg" alt="linear regression learning curve" height="300px" style="padding:10px">

1. If the _training cost_ is high and _validating cost_ is hight too, then it's a case of **High Bias** and $J_{train}$ is almost same as $J_{cv}$.

$$J_{train} \approx J_{cv}$$

2. If the _training cost_ is low and _validating cost_ is high, then it's a case of **High Variance**. $\gg$ (double greater than) means $J_{cv}$ is very very much high than $J_{train}$.

$$J_{cv} \gg J_{train}$$

3. If the _training cost_ is high and _validation cost_ is also high, even higher than the _training cost_, then it's a case of **High Bias** and **High Variance** (or **Low Variance** and **High Variance**). It happens sometimes in **neural networks**, where some part of the network is **underfit** and some part is **overfit** to the data.

<img src="./images/bias_and_variance.jpg" alt="bias and variance" height="400px" style="padding:10px">

---
