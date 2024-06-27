# Week 2: Recommender Systems

**Learning Objectives**
- Implement collaborative filtering recommender systems in TensorFlow
- Implement deep learning content based filtering using a neural network in TensorFlow
- Understand ethical considerations in building recommender systems

---

## Ch 1: Collaborative filtering

### Recommender Systems

A **Recommendation System** is a type of information filtering system which predicts the preferences of a user for specific items.
- It provides **personalized** recommendations to users based on their past preferences or other users' preferences for similar items.

> This is one of the topics that has received quite a bit of attention in academia. But the commercial impact and the actual number of practical use cases of recommended systems seems to me to be even vastly greater than the amount of attention it has received in academia. ~ _Andrew Ng_

#### Example: Movie Recommendation System

Say, we have some movies and some users who have rated these movies on a scale of $0 - 5$. We can represent this data in a matrix form where each row represents a movie and each column represents a user. 
The matrix (data) is sparse because not all users have rated all movies, meaning that there might be some missing values represented by $?$, which we need to predict.

| Movie        | User 1 | User 2 | User 3 | User 4 | User 5 |
| ------------ | :----: | :----: | :----: | :----: | :----: |
| *Avatar*       |   1    |   2    |   3    |   5    |   2    |
| *Brahmastra*   |   5    |   4    |   ?    |   0    |   0    |
| *Inception*    |   2    |   5    |   4    |   0    |   ?    |
| *Interstellar* |   4    |   ?    |   5    |   3    |   5    |
| *Titanic*      |   3    |   0    |   0    |   ?    |   4    |
| ...          |  ...   |  ...   |  ...   |  ...   |  ...   |

**Notations**:
- $n_u$ = number of users
- $n_m$ = number of movies
- $r(i, j) = 1$ if user $j$ has rated movie $i$ else $0$
- $y^{(i, j)}$ = rating given by user $j$ to movie $i$ (if rated)

##### Approach
One way we can approach this problem is to look at the movies that users have not rated and try to predict the ratings for those movies. We can then recommend the movies with the highest predicted ratings to the users.

---

### User-per Item Features

Let's say with different user ratings, we have some more features for each movie. For example: $x_1$ = Romance, $x_2$ = Action, etc... These features can be used to predict the ratings for the movies.

#### Data

| Movie        | User 1 | User 2 | User 3 | User 4 | User 5 | $x_1$ (Romance) | $x_2$ (Action) |
| ------------ | :----: | :----: | :----: | :----: | :----: | :-------------: | :------------: |
| *Avatar*       |   1    |   2    |   3    |   5    |   2    |      0.6       |      0.7       |
| *Brahmastra*   |   5    |   4    |   ?    |   0    |   0    |       0.8       |      0.9       |
| *Inception*    |   2    |   5    |   4    |   0    |   ?    |       0.4       |      0.9       |
| *Interstellar* |   4    |   ?    |   5    |   3    |   5    |       0.7       |      0.6       |
| *Titanic*      |   3    |   0    |   0    |   ?    |   4    |       0.9       |      0.2       |


#### Notations
- $n$ = number of features
- $x^{(i)}$ = feature vector for movie $i$
- $w^{(j)}, b^{(j)}$ = parameters for user $j$
- $r(i, j) = 1$ if user $j$ has rated movie $i$ else $0$
- $y^{(i, j)}$ = rating given by user $j$ to movie $i$ (if rated)
- $m^{(j)}$ = number of movies rated by user $j$


Here, we have $n=2$ features for each movie, where for each movie $i$, we have a feature vector $x^{(i)} \in \mathbb{R}^2$ and each feature is in the range $0-1$.

For movie $1$ - _Avatar_:

$$x^{(1)} = \begin{bmatrix} 0.6 \\ 0.7 \\ \end{bmatrix}$$

For movie $4$ - _Interstellar_:

$$x^{(4)} = \begin{bmatrix} 0.7 \\ 0.6 \\ \end{bmatrix}$$

#### Prediction

Now, to predict rating for a movie $i$ by user $j$, we can use the simple **Linear Regression** model:

$$f(x^{(i)})_{(w^{(j)}, b^{(j)})} = w^{(j)} \cdot x^{(i)} + b^{(j)}$$

where $x^{(i)}$ is the feature vector for movie $i$ and $w^{(j)}$ and $b^{(j)}$ are the parameters for user $j$.


Example: For user $3$ and movie $2$ - _Brahmastra_:

$$f(x^{(2)})_{(w^{(3)}, b^{(3)})} = w^{(3)} \cdot x^{(2)} + b^{(3)}$$

and say,

$$w^{(3)} = \begin{bmatrix} 0.9 \\ 0.3 \\ \end{bmatrix}, \quad b^{(3)} = 0.1$$

then,

$$f(x^{(2)})_{(w^{(3)}, b^{(3)})} = \begin{bmatrix} 0.9 \\ 0.3 \\ \end{bmatrix} \cdot \begin{bmatrix} 0.8 \\ 0.9 \\ \end{bmatrix} + 0.1 \\[1em] = 0.72 + 0.27 + 3 = 3.99$$

So, the predicted rating for user $3$ and movie $2$ _Brahmastra_ is $3.99$.

#### Cost Function

The cost function for this model can be the **Mean Squared Error**:

$$ J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:r(i, j)=1} \left( f(x^{(i)})_{(w^{(j)}, b^{(j)})} - y^{(i, j)} \right)^2 $$

which can be exapanded as:

$$ J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:r(i, j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 $$

Here, we are summing over all the movies rated by user $j$.

W can also add a regularization term - **L2 (*Ridge*) regularization** to the cost function to prevent overfitting. The cost function with regularization term is given by:

$$  J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:r(i, j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2m^{(j)}} \sum_{k=1}^n \left( w_k^{(j)} \right)^2 $$

where $\lambda$ is the regularization parameter and regularization is happening over all the parameters $w_k^{(j)}$, where $k=1, 2, \ldots, n$ (# features).

#### Modification for Multiple Users

Here, in the cost function, we need to do a little modification, i.e. rather than summing over all the movies rated by user $j$, we need to sum over all the users who have rated movie $i$.

Then the cost function for multiple users can be given by:

$$ J \begin{pmatrix} w^{(1)} & w^{(2)} & \cdots & w^{(n_u)} \\ b^{(1)} & b^{(2)} & \cdots & b^{(n_u)} \end{pmatrix}  = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i, j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 $$

And the regularization term can be added as:

$$ J \begin{pmatrix} w^{(1)} & w^{(2)} & \cdots & w^{(n_u)} \\ b^{(1)} & b^{(2)} & \cdots & b^{(n_u)} \end{pmatrix}  = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i, j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left( w_k^{(j)} \right)^2 $$


Now, instead of optimizing the parameters for each user separately, we can optimize all the parameters together using **Gradient Descent**.

