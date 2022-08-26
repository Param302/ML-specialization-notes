# Week 3: Advice for Applying Machine learning

## Machine Learning Development process

### Iterative loop of Machine Learning Development

Let's see what's the overall develpoment process of a **Machine Learning** model is often like:

<img src="./images/ml-development.jpg" alt="ml development process" width="600px" style="padding:50px">

#### Step 1: Choose Architecture

-   $1^{st}$ step is to choose the design of our model, the architecture, if it's a **neural network**, then how many _layers_, _neurons_ does it have.
-   And, we need to choose the portion of data which we are going to use, like how many features we are gonna use, should we do feature engineering or not or choose all of them.

#### Step 2: Train model

-   $2^{nd}$ step is to train our model.

#### Step 3: Diagnostics

-   $3^{rd}$ step is to diagnose our model, in which we see the learning curve, the loss, accuracy of our model and how it performs.
-   We check **Bias**, **Variance** and do _error analysis_.
-   Then, if we don't satisfy with the model's performance, then we need to repeat from [step $1$](#Step-1:-Choose-Architecture), i.e. should we add more data, or reduce the value of $\lambda$ or increase it, add more features, remove some features etc..., according to the situation we are in.

-   So, to conclude that, It is a loop where we do all these steps again and again until we get a satisfactory result from our model.

---

### Spam Email Classifier

Let's take an example of Spam email classification.

Let's say we have 2 different emails, where $1$ is spam and fake, $2^{nd}$ one is not spam.

| Spam email                                                                                                                                                               | Real email                                                                                                                                                                                                                                |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| From: cheapsales@buystufffromme.com<br/>To: Andrew Ng<br/>Subject: Buy now!<br/></br>Deal of the week! Buy now!<br/>Rolex w4tches - $100<br/>Medicine - (any kind) - $50 | From: Parampreet Singh <br/>To: Andrew Ng<br/>Subject: Thanks for your courses<br/></br>Dear Andrew sir,<br/>Thanks for all the courses you have made.<br/>Your courses are really amazing and learning a lot from it.<br/>Thank you</br> |

In the above table, we can see that

-   $1^{st}$ email is clearly spam, just by looking at it's email address we can identify, also it's content, and spammers now-a-days, intentionally misspell some words, so that it cannot be marked as spam.
-   $2^{nd}$ email is a real email, sent by me, (will sent it for sure after this course 😅), and this is clearly not a spam.

So, how can we make a ML model, which can identify, which one is spam which one isn't.

1. We can make a list of features (words) like a $10,000$ words dictionary.
2. From that dictionary, for each email, we can create a vector $\vec{x}$ and map $0$ if the word isn't in the email or $1$ if the word is present in the email.
3. Or, we can also map the count of word.
4. Example, we have a list of words like: _a, andrew, buy, deal, discount_ etc...
5. So, we can create a vector $\vec{x}$ and map it's count or absence with $0$.
6. Like, for the spam email, the vector $\vec{x}$ will be like this:
   $$\vec{x} = \begin{bmatrix}0 \\ 1 \\ 2 \\ 1 \\ 1 \\ \vdots\end{bmatrix}$$

7. For, real email, it can be like this:
   $$\vec{x} = \begin{bmatrix}0 \\ 1 \\ 0 \\ 0 \\ 0 \\ \vdots\end{bmatrix}$$

8. With these set of features, we can make our model using _logistic regression_ or a complex model by making a **neural network** which predicts $y$ spam or not, for the given input features(occurences of words) $\vec{x}$.

-   Let's say, after training the model, it doesn't work well, so how do we diagnose it ?

#### Diagonizing Spam Email classifier

We can try different ways from the following:

1. Collect more data.
2. Develop sophisticated features based on email routing (like header of email)
3. Define sophisticated features from email body like _discounting_ and _discount_ and _discounted_ should be treated as same.
4. Design algorithms to detect misspelling like w4tches etc...

Let's say, after training our model for the $1^{st}$, we got **High Bias**, i.e. our _training error_ is high as well as our \*_validation error_ is high too!

So, to tackle it, we cannot follow the option $1$ i.e. collect more data, because we saw earlier, that adding more data doesn't fix the **High Bias** problem, instead it fixes **High Variance** problem.

So, just by doing trial and error and doing this develpoment process in a loop for multiple times, we can made a good model.

---

### Error analysis

Let's say we have trained our model and predict $\hat{y}$ on _cross-validation_ dataset where $m_{cv} = 500$, out of which $100$ are missclassified.

So, in error analysis, what we do is, we manually examine a sample of that missclassified data, or all of them if it's small number, and categorize them based on common properties.

Like for Spam Email classification, we can categorize, missclassified examples into these $5$ properties/traits:

1. Pharama (medicine related)
2. Deliberate misspelling (w4tches)
3. Unusual email routing (email address, header)
4. Steal passwords (phishing)
5. Spam message in embedded image

So, say we have categorize $60$ misclassified examples out of $100$ as follows:
| Categories | no. of examples |
|:-----------|:---------------:|
| Pharama (medicine related) | $21$ |
| Deliberate misspelling (w4tches) | $12$ |
| Unusual email routing (email address, header) | $6$ |
| Steal passwords (phishing) | $7$ |
| Spam message in embedded image | $4$ |

There can be some emails, where they can be counted in more than one categories like some emails may have deliberate misspellings as well as unusual email routing.

So, we need to look out which category has more number of examples, and give first priority to it. It doesn't we just ignore the low counts one like spam message in embedded image, we just first need to focus on the emails which has the highly impacted cateogory like Pharama and deliberate misspellings.

**Error analysis** would really be useful, rather than just focusing randomly on one kind of example.

> By the way, I'm telling the story because I once actually spent a lot of time building algorithms to find deliberate misspellings and spam emails only much later to realize that the net impact was actually quite small. This is one example where I wish I'd done more careful error analysis before spending a lot of time myself trying to find these deliberate misspellings. ~ _Andrew Ng_

With **error analysis**, we can also find what's the problem and what approach should we try out from [it](#diagonizing-spam-email-classifier).

Like, $4^{th}$ point should be useful i.e. _Design algorithms to detect misspelling like w4tches etc..._ .

---

### Adding data

Sometimes, adding more data can help our machine learning model to improve it's performance, like when we have high variance problem.

Let's see different ways to add data for machine learning model, so that it can learn better.

#### 1. Adding more data

> Look up for new unlabelled data, labelled them and add them into _training_ dataset, especially, the type of data which is causing error which can be done by _error analysis_.

One way to add data, is to look up to new data, and add that examples to our dataset.

We can add that examples which are causing high error, by doing _error analysis_ we can find which type of examples our model is not classifying properly.

From the above _Spam Email classifier_ example, we saw that:

-   Out of $m_{cv} = 500$, _cross-validation_ examples, $100$ are missclassified.
-   And, from $100$ missclassified examples, we did _error analysis_ on $50$ of them
-   Out of $50$ majority of them belongs to _Pharma_ category.

So, what we can do is we can look up to new unlabelled data and add that data which belongs to _Pharama_ category.

By adding more data of missclassified examples to our training data, we can make our model learn that data as well.

---

#### 2. Data Augmentation

> It is the technique used to generate new _training_ examples from existing _training_ examples by modifying them.

It is mainly used in image and audio related problems.

In this technique, we increase the diversity of our _training_ set by apply realastic transformations.

Let's take _Hand written Alphabet recognition_.

-   Say, we have few images of letter _A_ compared to other letters.

-   We want to add more examples of _A_ in our _training_ set.
-   We can take that picture of _A_, and modify it little bit, increase/decrease the scale of image, rotate it a little bit, stretch it and make new examples of it.

<img src="./images/data-augmentation.jpg" alt="hand written alphabet 'a' data augmentation" width="500px" style="padding:10px">

Similarly, in _Speech recognition_, we can add new sounds in the background of voices, like noise of crowd, traffic jam, noise of people talking etc... and make new voice examples from them.

One important thing to be aware while doing _data augmentation_ is:

-   The changes which we are doing to the _training_ data, should be realastic.
-   They should be real life transformations, like rotating images, stretching images, not like adding noise in the alphabets, which can never be found in the _test_ data.
-   Purely random data transformations may lead to poorely trained model.

---

#### 3. Data synthesis

> In _data synthesis_, we use artificial data inputs to create new _training_ examples.

In _data augmentation_, we use existing data to create new data.

But, in _data synthesis_, we make new data which looks similar to the _training_ data.

Like, in **Photo OCR** (Photo Optical Character Recognition).

Say, we have this image, we need to recognize characters in it.

<img src="./iamges/../images/data-synthesis.jpg" alt="data synthesis example" width="400px" style="padding:10px">

Here, we can see a lot of characters in this photo.

<img src="./iamges/../images/data-synthesis-2.jpg" alt="data synthesis example 2" width="400px" style="padding:10px">

Say, we have collected all the photos together, and this is how our _training_ set looks like:

<img src="./iamges/../images/data-synthesis-3.jpg" alt="data synthesis example 3" width="400px" style="padding:10px">

Now, in _data synthesis_, what we do is:

-   We can see there are different kinds of fonts with different sizes used in these images.
-   So, we can make new data, just like it.
-   Like, we can write different characters in different font families, sizes and different colors.
-   And, take screenshots of them, or make images of them.
-   With this, we can create more _training_ data like this:

<img src="./iamges/../images/data-synthesis-4.jpg" alt="data synthesis example 4" width="800px" style="padding:10px">

-   Here, on the left, we have _real data_ and on right _Synthetic data_, which we have created ourselves.
-   It is not looking different from it, instead it looks very similar to the _real data_.
-   It may take a lot of time to make _synthetic_ data, but it can give a huge boost to our model's performance.

---

### Engineering the data used by our system

-   Most machine learning researchers attention was on the **Conventional Model Centric** approach.

$$\text{AI} = \underbrace{\text{Code }}_{\text{main focus}} + \text{ Data}$$

-   Where, their main focus is on code, i.e. the machine learning model, how to improve it, what techniques we can use to improve it's performance.
-   This is a good approach, will lead us to great machine learning algorithms like **linear regression, logistic regression and neural networks**.
-   But, there's also another another which is **Data centric** approach
-   In this approach, researchers mainly focused on the data engineering and make data useful for our machine learning algorithms.

$$\text{AI} = \text{Code } + \underbrace{\text{ Data}}_{\text{main focus}}$$

-   Sometimes, focus on data would be very efficient for model making process.
-   So, we should also spend some time with data.

---
