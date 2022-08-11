# Week 2: Neural Network Training

## Ch 3: Multiclass Classification

### What is Multiclass Classification?

> _Multiclass classification_ refers to the **classification** problems where you can have more than just two possible output labels so not just $0$ or $1$. ~ _Andrew Ng_

In _Multiclass classification_ problem, we have to classify the _target_ variable $y$ into multiple classes (more than $2$), whereas in _Binary classification_, we classify the target variable $y$ into two classes ($0$ or $1$).

---

#### Hand-written digit recognition

Earlier, in _Hand-written digit recognition_, we are classifying the target variable $y$ into $2$ classes $0$ or $1$, whether the digit is a zero or a one, which is considered as _Binary classification_ problem.

<img src="./images/binary_classification_example.jpg" alt="binary classification hand written digit example" width="200px" style="padding:10px">

Now, we will classify the _Hand-written digit_ into $10$ classes $0$ to $9$, which is _Multiclass classification_ problem.

<img src="./images/multiclass_classification_example.jpg" alt="multiclass classification hand written digit example" width="800px" style="padding:10px">

---

#### Graph Illustration

#### Binary Classification

In _Binary classification_ with $2$ _input_ features, graph looks like this:

<img src="./images/binary_classification.jpg" alt="binary classification graph" width="400px" style="padding:10px">

Where we need to find the probability of the target variable $y$ being $1$ i.e. $P\left(y=1|\vec{x}\right)$.

#### Multiclass Classification

In **Multiclass classification** with $2$ _input_ features, graph looks like this:

<img src="./images/multiclass_classification.jpg" alt="multiclass classification graph" width="400px" style="padding:10px">

Where we need to find probability of target variable $y$ being $1$, $2$, $3$ or $4$, i.e. $P\left(y=1|\vec{x}\right)$, $P\left(y=2|\vec{x}\right)$, $P\left(y=3|\vec{x}\right)$ and $P\left(y=4|\vec{x}\right)$.

And in the above graph, _decision boundary_ will look like this:

<img src="./images/multiclass_classification_2.jpg" alt="multiclass classification decision boundary graph" width="400px" style="padding:10px">

---

