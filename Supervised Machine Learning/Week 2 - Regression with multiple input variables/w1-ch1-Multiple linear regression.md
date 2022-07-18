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
| $w$ *coefficient* | $w$ *value* | $x$ *feature no.* | $x$ *feature*     | 
|:-----------------:|:-----------:|:-----------------:|:-----------------:|
| $w_1$             | `0.1`       | $x_1$             | *house size*      |
| $w_2$             | `4`         | $x_2$             | *no. of bedrooms* |
| $w_3$             | `10`        | $x_3$             | *no. of floors*   |
| $w_4$             | `-2`        | $x_4$             | *house size*      | 
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


### Notations in Machine learning
<img src="./images/multiple%20features.jpg" alt="multiple features of house data" width="600px">

| Notation | Meaning | Example |
|:--------:|:-------:|:-------:|
| x<sub>1</sub> | 1st feature of `x` | $x_1$ = *Size in feet<sup>2</sup>* |
| `j` | $j^{th}$ feature of training data | $x_2$ = *no. of bedrooms* |
| `n` | no. of features | `n = 4` |
| $\vec{x}$ | A vector (basically, a list of values) | $\vec{x} = [1, 2, 3, 4]$ |
| $\vec{x}^{(i)}$ | features of $i^{th}$ training data | $\vec{x}^{(2)} = [1416, 3, 2, 30]$ |
|  $\vec{x}^{(i)}_j$ | $j^{th}$ feature of $i^{th}$ training data | $\vec{x}^{(2)}_4 = 40$ |


----

### Quizzes

#### Video quiz 1

<img src="../quizzes/Video%20quiz%209%20-%20multiple%20linear%20regression.jpg" alt="video quiz 1" width="60%">

<details>
<summary><font size='3' color='#00FF00'>Answer to <b>Video quiz 1</b></font></summary>
<p>If you thought 852 then your are right! 852 is the first feature (first column in the table) of the fourth training example (fourth row in the table).</p>
</details>