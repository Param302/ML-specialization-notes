# Week 2 - Regression with mutliple input variables
**Learning Objectives:**
- Use vectorization to implement multiple linear regression
- Use feature scaling, feature engineering, and polynomial regression to improve model training
- Implement linear regression in code

---

## Ch 1: Multiple linear regression

### Multiple features
> Till now, we have worked only with a single feature in *linear regression*. But now we will learn how to work with *multiple features*. ~ *Andrew Ng*

Consider this example:

<img src="./images/multiple%20features.jpg" alt="multiple features of house data" width="600px">

Earlier, we only saw a single feature *house size* but now we have multiple features like:
- *house size*
- *no. of bedrooms*
- *no. of floors*
- *age of home*

So, now we will use all of these features to predict *price* of a house.

> *Linear regression* with multiple *features* also known as ***Multiple linear regression***. ~ *Andrew Ng*

#### Formula of *Linear Regression* model with *mutliple features*
> Previously, the formula for *linear regression* was: 
> $$f_{w,b}(x) = wx + b$$
> But, now with *multiple features* our new formula will be:
> $$f_{w,b}(x) = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + b$$
> Where, for each feature we will a specific `w` parameter. ~ *Andrew Ng*

Example:
$$f_{w,b}(x) = 0.1 x_1 + 4 x_2 + 10 x_3 + -2 x_4 + 80$$
where:
| $w$ *coefficient* | $w$ *value* | $x$ *feature no.* |   $x$ *feature*   |
| :---------------: | :---------: | :---------------: | :---------------: |
|       $w_1$       |    `0.1`    |       $x_1$       |   *house size*    |
|       $w_2$       |     `4`     |       $x_2$       | *no. of bedrooms* |
|       $w_3$       |    `10`     |       $x_3$       |  *no. of floors*  |
|       $w_4$       |    `-2`     |       $x_4$       |   *house size*    |
- `80` is base price
> If the model is trying to predict  the price of the house in thousands of dollars, you can think of this `b` equals `80` as saying that the base price of a house starts off at maybe `$80,000`, assuming it has no size, no bedrooms, no floor and no age. ~ *Andrew Ng*
> 
> You can think of `0.1` is for every additional *square foot*, so the price will increase by `0.1%` or `$100`. `4` for each *bedroom*. `10` for each *floor*. `-2` for *house size*, that means the price maybe decrease by `$2000`. ~ *Andrew Ng*

Let's rewrite the formula in a simple way:
- By defining $\vec{w}$ , as a vector (list of weights) like this:
$$\vec{w} = [w_1, w_2, w_3, \ldots w_n]$$
- $\vec{w}$ is a list of parameters for our *linear regression* function.
- Similarly, by defining $\vec{x}$ , as a vector (list of features) like this:
$$\vec{x} = [x_1, x_2, x_3, \ldots x_n]$$
- $\vec{x}$ is a list of features of our *training data*.
- So, our final formula will be:
> $$f_{\vec{w}, b} (\vec{x}) = \vec{w}\cdot\vec{x} +  b$$

Here $\cdot$ (dot) means *dot product*.
> In *dot product* of *vectors* we mutliply corresponding value of both *vectors* and then add up all of them, like this:
> $$\vec{w}\cdot\vec{x} = w_1x_1 + w_2x_2 + w_3x_3 + \ldots + w_nx_n$$


----

### Vectorization
> Vectorization makes code shorter, so hopefully easier to write and it also makes it run faster. ~ *Andrew Ng*

> A style in computer programming where operations are applied to whole arrays instead of individual elements. ~ [*Wikipedia*](https://en.wikipedia.org/wiki/Vectorization#:~:text=a%20style%20of%20computer%20programming%20where%20operations%20are%20applied%20to%20whole%20arrays%20instead%20of%20individual%20elements)

Starter code:
```python
import numpy as np
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```
- We are importing *numpy* library. *Numpy* is a numerical linear algebra library used to do mathematical calculations in **Python**.


#### *Linear Regression* function code without vectorization

1. using normal calculation
```python
f = w1 * x1 + w2 * x2 + w3 * x3 + b
```
- Here, we are writing each weight and it's corresponding feature manually, it is easier to write if we have `3-4` features, but what if we have `100` or `10,000` features, it will take a lot of time to write.

2. using loop
```python
f = 0
for j in range(n):  # n = no. of features
    f += w[j] * x[j]
f += b
```
- With this way, we don't have to write manually each weight and feature's value, but loop takes some time, which makes it less efficient.

#### *Linear Regression* function code with vectorization
```python
f = np.dot(w, x) + b
```
or
```python
f = w @ x + b
```
- Both codes above are doing same thing, they are calculating dot product using vectorization, which is fast, because *numpy* library is doing calculations in `C` language under the hood. And the code is  also short.
- *Numpy* do *dot product* parallely at same time for all values of both numpy arrays.
- `@` in 2nd code, means dot product.

**Note:** Use `@` only with *numpy **arrays***.

Imagine, you have `16` features in your dataset, and you want to do *Gradient Descent* to calculate best values for all `w`'s and `b` parameter.

So, you will calculate derivative for each `w` and store it in $\vec{d}$ vector like this:
$$\vec{w} = [w_1, w_2, w_3, \ldots w_n]$$
$$\vec{d} = [d_1, d_2, d_3, \ldots d_n]$$
And, learning rate is `0.1`.

Now, you want to compute `w` for each feature, without vectorization, your code will look like this:
```python
for j in range(16):
    w[j] -= 0.1 * d[j]
```
- It is using loop, so you don't have to write much code manually, it is not efficient.

If you use vectorization to do this, your code will be this:
```python
w = w - 0.1 * d
```
- With vectorization, under the hood, it multiplies all the values of `d` with `0.1` and then subtract with all values of `w` parallely, which makes it very efficient.

----

### Jupyter lab [optional] [🔗](../codes/W2%20-%20L1%20-%20Numpy%20Vectorization.ipynb)

----

### *Gradient Descent* for multiple linear regression

Let's see what is the formula for *multiple linear regression* for *Gradient Descent*:

- Earlier, for one feature our formula was:
$$repeat\enspace until\enspace convergence \{ $$
$$w = w - a\frac{1}{m}\sum\limits_{i=1}^m(f_{w,b}(x^{(i)} - y^{(i)})x^{(i)}$$
$$b = b - a\frac{1}{m}\sum\limits_{i=1}^m(f_{w,b}(x^{(i)} - y^{(i)})$$
$$ \} $$

- But now, for multiple features our formula is:
$$repeat\enspace until\enspace convergence \{ $$
$$w_1 = w_1 - a\frac{1}{m}\sum\limits_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)} - y^{(i)})x_1^{(i)}$$
$$ \vdots $$
$$w_n = w_n - a\frac{1}{m}\sum\limits_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)} - y^{(i)})x_n^{(i)}$$
$$b = b - a\frac{1}{m}\sum\limits_{i=1}^m(f_{w,b}(x^{(i)} - y^{(i)})$$
$$ \} $$

- Now, for each feature $j$, gradient descent will find best $w_j$ parameter. So, it will run for total of $j$ features + $b$ paramter.


----

### Normal equation method
> We know that *gradient descent* is a great method for minimizing the cost function $j$ to find $w$ and $b$, but there is one another algorithm that works only for *linear regression* which find *coefficients* all in one goal without iterations. This method is called **normal equation* method. ~ *Andrew Ng*

But, it has some disadvantages:
- It doesn't generalize to other learning algorithms like logistic regression etc...
- It runs slow when number of features are more than $10^4$ ($10,000$).


----

### Jupyter lab [optional] [🔗](../codes/W2%20-%20L2%20-%20Multiple%20Variable%20Linear%20Regression%20[optional].ipynb)

----

### Notations in Machine learning
<img src="./images/multiple%20features.jpg" alt="multiple features of house data" width="600px">

|     Notation      |                  Meaning                   |              Example               |
| :---------------: | :----------------------------------------: | :--------------------------------: |
|   x<sub>1</sub>   |             1st feature of `x`             | $x_1$ = *Size in feet<sup>2</sup>* |
|        `j`        |     $j^{th}$ feature of training data      |     $x_2$ = *no. of bedrooms*      |
|        `n`        |              no. of features               |              `n = 4`               |
|     $\vec{x}$     |   A vector (basically, a list of values)   |      $\vec{x} = [1, 2, 3, 4]$      |
|  $\vec{x}^{(i)}$  |     features of $i^{th}$ training data     | $\vec{x}^{(2)} = [1416, 3, 2, 30]$ |
| $\vec{x}^{(i)}_j$ | $j^{th}$ feature of $i^{th}$ training data |       $\vec{x}^{(2)}_4 = 40$       |


----

### Quizzes

#### Video quiz 1
<img src="../quizzes/Video%20quiz%209%20-%20multiple%20linear%20regression.jpg" alt="video quiz 1" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>Video quiz 1</b></font></summary>
<p>If you thought 852 then your are right! 852 is the first feature (first column in the table) of the fourth training example (fourth row in the table).</p>
</details>

#### Video quiz 2
<img src="../quizzes/Video%20quiz%2010%20-%20multiple%20linear%20regression.jpg" alt="video quiz 2" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>Video quiz 2</b></font></summary>
<p>If you have selected 1st option then your are right! This numpy function uses parallel hardware to efficiently calculate the dot product.</p>
</details>

#### Video quiz 3
<img src="../quizzes/Video%20quiz%2011%20-%20multiple%20linear%20regression.jpg" alt="video quiz 3" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>Video quiz 3</b></font></summary>
<p>If you have selected 1st option then your are right!</p>
</details>

#### Practice quiz 1 question 1
<img src="../quizzes/Quiz%20-%204%20Multiple%20linear%20regression%20q1.jpg" alt="practice quiz 1 question 1" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>question 1</b></font></summary>
<p>If you thought 30 then your are right! It is the 4th feature (4th column in the table) of the 3rd training example (3rd row in the table).</p>
</details>

#### Practice quiz 1 question 2
<img src="../quizzes/Quiz%20-%204%20Multiple%20linear%20regression%20q2.jpg" alt="practice quiz 1 question 2" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>question 2</b></font></summary>
<p>If you have selected d option (All of the above) then your are right! All of these are benefits of vectorization!</p>
</details>

#### Practice quiz 1 question 3
<img src="../quizzes/Quiz%20-%204%20Multiple%20linear%20regression%20q3.jpg" alt="practice quiz 1 question 3" width="60%">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>question 3</b></font></summary>
<p>If you have think it's False then your are right! Doubling the learning rate may result in a learning rate that is too large, and cause gradient descent to fail to find the optimal values for the parameters w and b.</p>
</details>