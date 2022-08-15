# Week 3: Advice for Applying Machine Learning
**Overview:**
- Evaluate and then modify your learning algorithm or data to improve your model's performance
- Evaluate your learning algorithm using cross validation and test datasets.
- Diagnose bias and variance in your learning algorithm
- Use regularization to adjust bias and variance in your learning algorithm
- Identify a baseline level of performance for your learning algorithm
- Understand how bias and variance apply to neural networks
- Learn about the iterative loop of Machine Learning Development that's used to update and improve a machine learning model
- Learn to use error analysis to identify the types of errors that a learning algorithm is making
- Learn how to add more training data to improve your model, including data augmentation and data synthesis
- Use transfer learning to improve your model's performance.
- Learn to include fairness and ethics in your machine learning model development
- Measure precision and recall to work with skewed (imbalanced) datasets

---

## Ch 1: Advice for Applying Machine Learning

### Debugging a learning algorithm
Let's say we have made a **Regularized** *Linear Regression* model, and it's cost function is:
$$J(\vec{w}, b) = \frac{1}{2m}\sum_{i=1}^m\left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m}\sum^n_{j=1}w^2_j$$

But, while doing predictions with this model, it is giving unacceptably large errors.

So, what approaches we can try?
1. Get more training examples
2. Try smaller sets of features
3. Try getting additional features
4. Try adding polynomial features ($x^2_1, x^2_2, x_1, x_2$ etc...)
5. Try decreasing $\lambda$
6. Try increasing $\lambda$

---

### Evaluating a model
Model Evaluating is very important.

We can evaluate it's performance to check how well it's performing.

Let's take an example of *House price prediction*, where we are trying to predict the price of a house, and we have $4$ input features:
1. $x_1$: size in feet$^2$
2. $x_2$: number of bedrooms
3. $x_3$: number of floors
4. $x_4$: age of house in years
- $y$: price of house in $\$1000$

We are using *linear regression* algorithm to predict *house prices*, and we are using *polynomial features* till degree of $n$, i.e. from $w_1x$ to $w_4x^4$.

$$f_{\vec{w}, b}(\vec{x}) = w_1x + w_2x^2 + \cdots + w_nx^n + b$$

And, after fitting the *training data*, it's graph looks like this:

<img src="./images/house_price_graph.jpg" alt="house price graph" width="300px" style="padding:10px">

Here, we can see that:
- Model fits the *training data* so well, that it's fail to *generalize* to new data.

In order to see our model's performance, we need to evaluate it, by visualizing the new data points and comparing them with the actual prices.

We can do this by splitting our *house price* data into *training* and *testing* data.

---

#### Splitting the data

#### Regression problem
Say, we have $10$ data points here:

<img src="./images/house_price_dataset.jpg" alt="house price dataset" height="300px" style="padding:10px">

- We'll split our data into $2$ parts, say $70\%$ to *training* dataset and $30\%$ to *testing* dataset.

<img src="./images/house_price_splitted_dataset.jpg" alt="house price dataset" height="300px" style="padding:10px">

- And, we'll train our model on $70\%$ *training* dataset and evaluating it's performance on $30\%$ *testing* dataset.

- In notation, we'll use $m_{train}$ to signify *training* examples like $x^{m_{train}}$ & $y^{m_{train}}$ and $m_{test}$ to signify *testing* examples like $x^{m_{test}}$ & $y^{m_{test}}$.

- To train our model, we'll fit the *training* dataset with the *polynomial* features as shown above and minimize it's cost with *cost function* $J(\vec{w}, b)$.
$$J(\vec{w}, b) = \frac{1}{2m_{train}}\sum^{m_{train}}_{i=1}\left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m_{train}}\sum^n_{j=1}w^2_j$$

- To find the *training* error, we'll use same *cost function* $J(\vec{w}, b)$ on $m_{train}$ training examples, except we won't use *regularization* term.
$$J_{train}(\vec{w}, b) = \frac{1}{2m_{train}}\left[\sum^{m_{train}}_{i=1}\left(f_{\vec{w}, b}(\vec{x}^{(i)}_{train}) - y^{(i)}_{train}\right)^2\right]$$

- And, To compute the *test* error, we'll use same *cost function* $J(\vec{w}, b)$ on $m_{test}$ testing examples as above.
$$J_{test}(\vec{w}, b) = \frac{1}{2m_{test}}\left[\sum^{m_{test}}_{i=1}\left(f_{\vec{w}, b}(\vec{x}^{(i)}_{test}) - y^{(i)}_{test}\right)^2\right]$$

- So, as we saw that, on *training* dataset, our model performs well.

<img src="./images/house_price_graph.jpg" alt="house price graph" width="400px" style="padding:10px">

- But, when we plot *testing* dataset, we can see that there are large errors.

<img src="./images/house_price_graph_test.jpg" alt="house price graph" width="400px" style="padding:10px">

- On *training dataset*, we can see the *error* is very low, almost $0$, but on *testing dataset*, we can see the *error* is very-very high.

With this way, we can evaluating our model, by observing that our model is performing well on *training dataset*, but not *generalizing* on new data, i.e. *testing dataset*.

---

#### Classification problem
On *classification* problem, we will do same as above, minimzing the *cost function* $J(\vec{w}, b)$ on $m_{train}$ training examples, and evaluating it's performance on $m_{test}$ testing examples.

- As we know, our *cost function* is:
$$J(\vec{w}, b) = -\frac{1}{m}\sum^m_{i=1}\left[y^{(i)}\log\left(f_{\vec{w}, b}(\vec{x}^{(i)}\right) + \left(1 - y^{(i)}\right)\log\left(1 - f_{\vec{w}, b}(x^{(i)}\right)\right] + \frac{\lambda}{2m}\sum^n_{j=1}w^2_j$$

- For computing error on *training dataset*, we'll use same *cost function* $J(\vec{w}, b)$ on $m_{train}$ training examples, except we won't use *regularization* term.

$$J_{train}(\vec{w}, b) = -\frac{1}{m_{train}}\sum^{m_{train}}_{i=1}\left[y^{(i)}_{train}\log\left(f_{\vec{w}, b}(\vec{x}^{(i)}_{train}\right) + \left(1 - y^{(i)}_{train}\right)\log\left(1 - f_{\vec{w}, b}(x^{(i)}_{train}\right)\right]$$

- For computing error on *testing dataset*, we'll use same *cost function* $J(\vec{w}, b)$ on $m_{test}$ testing examples as above.
$$J_{test}(\vec{w}, b) = -\frac{1}{m_{test}}\sum^{m_{test}}_{i=1}\left[y^{(i)}_{test}\log\left(f_{\vec{w}, b}(\vec{x}^{(i)}_{test}\right) + \left(1 - y^{(i)}_{test}\right)\log\left(1 - f_{\vec{w}, b}(x^{(i)}_{test}\right)\right]$$

- But, there's another way to find error on *training* and *testing* dataset is to find out the **fraction** of the *test* and *train* set which is miss-classified.
$$hat{y} = \begin{cases}1 \quad\text{if }f_{\vec{w}, b}(x^{(i)}) \ge 0.5 \\ 0 \quad\text{if }f_{\vec{w}, b}(x^{(i)}) < 0.5\end{cases}$$

- And, we'll count $\hat{y} \neq y$ as error.
- where:
- - $J_{train}(\vec{w}, b)$ is the *fraction* of the *training* set that has been miss-classified.
- - $J_{test}(\vec{w}, b)$ is the *fraction* of the *testing* set that has been miss-classified.
  
---

