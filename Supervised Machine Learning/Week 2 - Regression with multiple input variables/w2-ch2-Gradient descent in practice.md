# Week 2 - Regression with multiple input variables

## Ch 2: Gradient Descent in practice

### Feature scaling

> Feature scaling is performing some transformation to data so tha features of data will range of 0 to 1, rather than different ranges. ~ _Andrew Ng_

> Feature scaling is a method used to normalize the range of independent variables or features in data. ~ [_Wikipedia_](https://www.datacamp.com/freeweek)

#### When feature scaling is to be done

> Feature scaling is done when you have different features that take on very different range of values, it can cause gradient descent to run slowly but rescaling the different features so that they all take on comparble range. ~ _Andrew Ng_

Let's see how feature scaling helps

1. Assume we have house price data with 2 features _size_ $x_1$ and _no. of bedrooms_ $x_2$. And _size_ ranges from `300` to `2000` and _no of bedrooms_ ranges from `0` to `5`.

$$ŷ = w_1x_1 + w_2x_2 + b$$

2. Let's take an example data where $x_1 = 2000$ and $x_2 = 5$ and price = `500k`.
3. Now, take a possible value of $w_1 = 50$ , $w_2 = 0.1$ and $b = 50$.
4. So, in this case, estimated price will be:
   $$ŷ = 50 * 2000 + 0.1 * 5 + 50$$
   $$ŷ = 100,000 + 0.5 + 50$$
   $$ŷ = 100,050.5k$$

5. `100,050.5k` is very large compared to actual price.
6. Let's take another set of possible values where, $w_1 = 0.1$ , $w_2 = 50$ and $b = 50$.
7. So, in this case, estimated price will be:
   $$ŷ = 0.1 * 2000 + 50 * 5 + 50$$
   $$ŷ = 200 + 250 + 50$$
   $$ŷ = 500k$$

8. Now, this estimated price is same as actual price.
9. So, here you might have notice that when a possible range of values of feature is large, like size of house, model will choose a relatively small parameter like `0.1`.
10. And, when range of values of feature is small, like no. of bedrooms, model will choose a relatively large parameter like `50`.

-   Now, let's see how these small and large range of values of features relate to _gradient descent_.

-   If you plot features of training data in a _scatter plot_, the graph will look like this:

<img src="./images/training-data-scatter-plot.jpg" alt="training-data-scatter-plot" width="500px">

-   And, the graph of model parameters on a _contour plot_ will look like this:

<img src="./images/training-data-contour-plot.jpg" alt="training-data-contour-plot" width="500px">

-   Now, you can see that _contour plot_ is so much stretched vertically and thin horizontally.
-   So, gradient descent may end up bouncing back and forth a long time before it reaches to the _global minimum_.

<img src="./images/training-data-contour-plot-2.jpg" alt="training-data-contour-plot-2" width="500px">

-   which means, gradient descent will take a lot of time.
-   So, to avoid this kind of situation we do _feature scaling_.
-   Now, if we scaled our both features _size of house_ and _no. of bedrooms_ in the range of values from `0` to `1`.
-   Our training data _scatter plot_ graph will look like this:

<img src="./images/training-data-scatter-plot-rescaled.jpg" alt="training-data-scatter-plot-rescaled" width="500px">

-   Now, our graph is looking like much more distributed than before.
-   So, because of _feature scaling_ our _contour plot_ will look like this:

<img src="./images/training-data-contour-plot-rescaled.jpg" alt="training-data-contour-plot-rescaled" width="500px">

-   It is like cocentric circles, rather than stretched ovals.
-   So, now _gradient descent_ can find the _global minimum_ easily by finding a direct path. And it also take less time to find.

---

#### Methods to do Feature scaling

| S.no. |                    Method                    |           Formula           | Explanation                                                                                                                                                    | Example                                                                                                                                                                                            |                                                   Graph                                                    |
| :---: | :------------------------------------------: | :-------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------: |
|  1.   |              Dividing by _max_               |       $\frac{x}{max}$       | Dividing each value by the maximum value of feature. It will mostly normalize values between `0` to `1`.                                                       | Let's say our house size is ranges from `300` to `2000`. So we will divide each value by `2000` and normalize it. <br/> After rescaling, now house size's value will range between `0.15` and `1`. |               <img src="./images/feature-scaling-maximum.jpg" alt="feature-scaling-maximum">               |
|  2.   |             _Mean normalization_             | $\frac{x - \mu}{max - min}$ | First, we will calculate mean of the range i.e. $\mu$ and subtract it from each value and then dividing result by subtracting minimum value by maximum value.  | Let's say no. of bedrooms in a house ranges from `0` to `5`. <br/> After applying _mean normalization_, we will get values of range `-0.46` to `0.54`.                                             |    <img src="./images/feature-scaling-mean-normalization.jpg" alt="feature-scaling-mean-normalization">    |
|  3.   | _Z-score normalization_ or _Standardization_ |  $\frac{x - \mu}{\sigma}$   | First, we will calculate mean of the range i.e. $\mu$ and subtract it from each value and then divide result by standard deviation of the range i.e. $\sigma$. | Let's say our house size is ranges from `300` to `2000`. <br/> After applying _z-score normalization_, it's will ranges from `-0.67` to `3.1`.                                                     | <img src="./images/feature-scaling-z-score-normalization.jpg" alt="feature-scaling-z-score-normalization"> |

---


### Quizzes

#### Video quiz 1
<img src="../quizzes/Video%20quiz%2012%20-%20feature%20scaling.jpg" alt="Video quiz 1" width="600px">
<details>
<summary><font size='3' color='#00FF00'>Answer to <b>video quiz 1</b></font></summary>
<p>If you have selected option a (Dividing each value by maximum of the value) then you are right! By dividing all values by the maximum, the new maximum range of the rescaled features is now 1 (and all other rescaled values are less than 1).</p>
</details>