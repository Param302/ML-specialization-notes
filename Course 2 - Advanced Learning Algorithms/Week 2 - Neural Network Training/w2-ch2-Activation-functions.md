# Week 2: Neural Network Training

# Ch 2: Activation Functions

Till now, we've been using _sigmoid_ function as our _activation function_ in **neural networks** for each unit.

But _sigmoid_ function is not suitable for every feature.

### T-shirt Demand prediction

Let's take an example of _T-shirt Demand prediction_.

where, we have $4$ features and $1$ target:

-   $x_1$ price
-   $x_2$ shipping cost
-   $x_3$ marketing
-   $x_4$ material
-   $y$ whether t-shirt is in demand $1$ or not $0$.

And, it's **neural network** architecture is:

<img src="./images/demand_prediction_nn.jpg" alt="demand prediction neural network" width="400px" style="padding:10px 50px">

where, we have:

-   $1$ hidden layer with $3$ neurons
-   -   $1^{st}$ neuron computes _affordability_ of t-shirt
-   -   $2^{nd}$ neuron computes _awareness_ of t-shirt
-   -   $3^{rd}$ neuron computes _perceived quality_ of t-shirt
-   $1$ output neuron computes _prediction_ of t-shirt demand

We can use _sigmoid_ function as activation function for _affordability_ and _perceived quality_, because either t-shirt is affordable or not, and it's upto their quality or not.

But in _awareness_, the user maybe not aware, or some aware or they knows about t-shirt very well. So, here _sigmoid_ function is not suitable, because it just results either $0$ or $1$, but it can be in a big range of values from $0$ to maybe $10$ or a much bigger number.

So, let's see some alteranatives of _sigmoid_ function.

---

### Alternatives to the Sigmoid activation function

We know that our *sigmoid function* gives probability of target and after applying threshold, we can get either $0$ or $1$.

<img src="./images/sigmoid_activation_function.jpg" alt="sigmoid activation function graph" width="400px" style="padding:10px 50px">

where:
- $z$ is dot product of input and weights and added bias or *linear regression* function
$$z = \vec{w} \cdot \vec{x} + b$$
- g(z) is the *sigmoid* function
$$g(z) = \frac{1}{1 + e^{-z}}$$
- If $g(z)$ is less than $0.5$, then $\hat{y} = 0$
- If $g(z)$ is equal to or greater than $0.5$, then $\hat{y} = 1$.


---

#### Rectified Linear Unit (ReLU)
We can use _ReLU_ function as *activation* function which gives the output from $0$ to a very large positive number.

<img src="./images/relu_activation_function.jpg" alt="relu activation function graph" width="400px" style="padding:10px 50px">

where:
- $z$ is same as in *sigmoid function*
$$z = \vec{w} \cdot \vec{x} + b$$
- Rather than using *sigmoid* function as $g(z)$, we will find max of $0$ and $z$, whichever is max, is the output.
$$g(z) = \max(0, z)$$
- If $z$ is negative, then $g(z)$ is $0$.
- If $z$ is $0$ or positive, then $g(z)$ is $z$.

#### Linear Activation function
There is another *activation* function known as *Linear activation* function.

<img src="./images/linear_activation_function.jpg" alt="linear activation function graph" width="400px" style="padding:10px 50px">

Which is just $z$,
where:
- $z$ is same as in *sigmoid function*
$$z = \vec{w} \cdot \vec{x} + b$$
- Rather than applying some formula to $z$, it just outputs $z$, that's why it results in single straight line, with an angle of $45\degree$
$$g(z) = z$$

- Sometimes, we often call it as **no** *activation* function, because it's just resulting $z$.
- So, whenever we hear that *no activation* function is used, means we are using *linear activation function* which just returns $z$.

---

