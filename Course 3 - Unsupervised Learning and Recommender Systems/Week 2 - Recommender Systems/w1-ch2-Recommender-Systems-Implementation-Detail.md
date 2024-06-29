# Week 2: Recommender Systems

## Ch 2: Recommender Systems Implementation Detail

### Mean Normalization

#### The Problem

Suppose, in our dataset, we have a new user who has not rated any of the movies, so all values are $?$ by default. We need to predict the ratings for this user.

| Movie          | User 1 | User 2 | User 3 | User 4 | User 5 | User 6 |
| -------------- | :----: | :----: | :----: | :----: | :----: | :----: |
| *Avatar*       |   1    |   2    |   3    |   5    |   2    |   ?    |
| *Brahmastra*   |   5    |   4    |   ?    |   0    |   0    |   ?    |
| *Inception*    |   2    |   5    |   4    |   0    |   ?    |   ?    |
| *Interstellar* |   4    |   ?    |   5    |   3    |   5    |   ?    |
| *Titanic*      |   3    |   0    |   0    |   ?    |   4    |   ?    |


Now, by using our **Linear Regression** model and **Cost Function** with **L2 Regularization**, we can predict the ratings for the new user.

Let's say for new user with $n=2$ features, we got the parameters as follows:

$$ w^{(6)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \hspace{1em} b = 0 $$

And if we predict the ratings for the new user, we get $\hat{y}^{(6)} = 0$ for all movies $i$, which is ofcrse not an ideal rating we should get.

#### Solution

To solve this problem, we can normalize the dataset, basically center our dataset around zero, which is called **Mean Normalization**.

In **Mean Normalization**, we subtract the mean of the dataset from each value in the dataset. So, the mean of the dataset becomes zero.

Here, we want to predict movie rating for a user $j$, so we will do the **Mean Normalization** for each of the user $j \in n_j$.

##### Step 1: Create the data matrix
We can take all the values in a matrix:

$$ \begin{bmatrix} 1 & 2 & 3 & 5 & 2 & ? \\ 5 & 4 & ? & 0 & 0 & ? \\ 2 & 5 & 4 & 0 & ? & ? \\ 4 & ? & 5 & 3 & 5 & ? \\ 3 & 0 & 0 & ? & 4 & ? \end{bmatrix} $$

##### Step 2: Calculate the mean of each rows i.e. for each user $j$

$$ \mu_j = \frac{1}{n_j} \sum_{i=1}^{n_j} x_{ij} $$

$$ \vec{\mu} = \begin{bmatrix} 2.6 \\ 2.25 \\ 2.75 \\ 4.25 \\ 2.33 \end{bmatrix} $$

##### Step 3: Subtract the mean from each value in the dataset

$$ x_{ij} = x_{ij} - \mu_j $$

$$ \begin{bmatrix} -1.6 & -0.6 & 0.4 & 2.4 & -0.6 & ? \\ 2.75 & 1.75 & ? & -2.25 & -2.25 & ? \\ -0.75 & 2.25 & 1.25 & -2.75 & ? & ? \\ -0.25 & ? & 0.75 & -1.25 & 0.75 & ? \\ 0.67 & -2.33 & -2.33 & ? & 1.67 & ? \end{bmatrix} $$

Now, even with the same parameters $w^{(6)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ and $b = 0$ on our **normalized** dataset, if we predict the ratings for the new user, we get $\hat{y}^{(6)} = 0$.

**But**, as we have subtracted the mean from the dataset, we have to **add** the mean to the predicted ratings to get the actual ratings.

$$ \hat{y}^{(6)} = \hat{y}^{(6)} + \mu_j $$

So, the actual ratings for the new user will be:

$$ w^{(6)} \cdot x^{(6)} + b = 0 + \begin{bmatrix} 2.6 \\ 2.25 \\ 2.75 \\ 4.25 \\ 2.33 \end{bmatrix} = \begin{bmatrix} 2.6 \\ 2.25 \\ 2.75 \\ 4.25 \\ 2.33 \end{bmatrix} $$

Now, these ratings are more meaningful than the previous ratings we got.

_Also, by **normalizing** the optimization algorithm will converge faster._

##### Alternate way
We can also normalize the dataset column-wise, i.e. for each movie $i$.
But in this case, this wouldn't be much helpful, as we are predicting the movie rating for each user.

---

### Tensorflow Implementation of Collaborative Filtering

#### Step 1: Load the Required Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
```

We will be using:
- `numpy` for numerical operations.
- `pandas` to load the dataset.
- `tensorflow` to implement the Collaborative Filtering.

#### Step 2: Create the Data Matrix

```python
data = pd.read_csv("movie_ratings.csv", index_col="movie")
```

- The dataset is loaded in the DataFrame `data`.

#### Step 3: Initialize required variables and Parameters

```python
n_movies, n_users = data.shape
n_features = 10

y = data.values
# R(i, j) = 1 if user j has rated movie i, else 0
R = np.where(y != 0, 1, 0)

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((n_users,  n_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((n_movies, n_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1, n_users),   dtype=tf.float64),  name='b')

learning_rate = 0.01
epochs = 100
lambda_ = 1
```

Here, we have initialized various variables:
- `n_movies` and `n_users` are the number of movies and users in the dataset.
- `n_features` is the number of features we want to use.
- `y` is the dataset values.
- `R` is the matrix where $R(i, j) = 1$ if user $j$ has rated movie $i$, else $0$.

Then, we have initialized the parameters:
- `W` is the matrix of shape $(n_{u}, n)$.
- `X` is the matrix of shape $(n_{m}, n)$.
- `b` is the bias term of shape $(1, n_{u})$.

Also, some additional variables like `learning_rate`, `epochs`, and `lambda_` are initialized.

#### Step 4: Initialize the Optimizer

```python
optimizer = tf.optimizers.Adam(learning_rate)
```

We have initialized the `Adam` optimizer with the `learning_rate` defined earlier.

**Adam** is an optimization algorithm that can be used instead of the classical **Gradient Descent** procedure to update network weights iterative based in training data. It's better than **Gradient Descent**.


#### Step 5: Define the Cost Function

```python
def cofi_cost_function(X, W, b, Y, R, n_users, n_movies, lambda_):
    # Predict the Ratings
    predictions = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    # Calculate the Cost
    cost = 0.5 * tf.reduce_sum(R * tf.square(predictions - Y))
    
    # Add Regularization
    cost += (lambda_ / 2) * (tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(X)))
    
    return cost
```

Here, we have defined the cost function for the Collaborative Filtering.


#### Step 6: Normalize the Data

```python
Y_mean = np.mean(y, axis=1)
Y_norm = y - Y_mean[:, np.newaxis]
```

We have calculated the mean of the dataset and subtracted it from the dataset to normalize the dataset.

#### Step 5: Run the Optimization

```python
for epoch in range(epochs):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:
        cost = cofi_cost_function(X, W, b, Y_norm, R, n_users, n_movies, lambda_)
        
    # Calculate the Gradients
    gradients = tape.gradient(cost, [W, X, b])
    
    # Update the Parameters
    optimizer.apply_gradients(zip(gradients, [W, X, b]))
```

Here, we have run the optimization for `epochs` number of times.

- We have used `GradientTape` to record the operations used to compute the cost.
- Then, we have calculated the gradients of the cost with respect to the parameters using `tape.gradient`.
- `tape.gradient` automatically calculates the gradients of the cost with respect to the parameters.
- Finally, we are applying the gradients to the parameters using the optimizer and updating them using `optimizer.apply_gradients`.

#### Step 6: Predict the Ratings

```python
predictions = tf.linalg.matmul(X, tf.transpose(W)) + b

# Add the Mean to the Predicted Ratings
predictions += Y_mean[:, np.newaxis]
```

Now, we are predicting the ratings for the users and adding the mean to the predicted ratings to get the actual ratings.

---


