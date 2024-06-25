# Week 1: Unsupervised Learning

## Ch 2: Anomaly Detection

### What is Anomaly Detection?

> Play video for highlighted transcript with text, Anomaly detection algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or an anomalous event., marked from 0 hours 0 minutes 6 seconds until 0 hours 0 minutes 20 secondsAnomaly detection algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or an anomalous event. ~ _Andrew Ng_

Anomaly detection is the process of identifying unexpected items or events in data sets, which differ from the normal behavior.

#### What is an _Anomaly_?

An anomaly is something that deviates from what is normal or expected.

Anomalies are also referred to as _outliers_, _novelties_, _noise_, and _deviations_.

#### Applications of Anomaly Detection

##### Fraud Detection
$x^{(i)}$ = features of user $i$'s activities
- How often user logs in?
- How often user logs in from different IP addresses?
- How many web pages visited?
- What kind of content user is interacting with in Social Media?
  
Model $p(x)$, and if $p(x) < \epsilon$, then flag as an anomaly, like unusual login activity or unusual content interaction.

##### Manufacturing
$x^{(i)}$ = features of machine $i$'s operations
- Detecting defects in aircraft engines
- Detecting defects in automotive parts
- Detecting defects in pharmaceuticals

Model $p(x)$, and if $p(x) < \epsilon$, then flag as an anomaly, like unusual engine temperature or unusual vibration.

##### Monitoring Computers in a Data Center
$x^{(i)}$ = features of machine $i$
- Memory use
- CPU load
- Network traffic
- Disk I/O

Model $p(x)$, and if $p(x) < \epsilon$, then flag as an anomaly, like unusual memory use or unusual network traffic or high CPU load.

#### Example: Aircraft Engine Anomaly Detection

Let's say we have a dataset of aircraft engine data, and we want to detect anomalies in the data.
Some features of the data are:
- $x_1$ = heat generated
- $x_2$ = vibration intensity

We have a dataset like this:
$$ \text{Dataset} = \{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\} $$

which is plotted as:

<img src="images/aircraft-engine-data.png" alt="Aircraft Engine data" width="500"/>

Here, if we see the data, we can see that most of the data points are clustered around a certain region, and there are a few data points that are far away from the cluster. The **<font color="#008BFF">blue</font>** circled data points are anomalies.

<img src="images/aircraft-engine-anomaly.png" alt="Aircraft Engine anomaly" width="500"/>

### Density Estimation

**Density estimation** is the process of estimating the **probability density function (PDF)** of the data.

Here, we can create a **Model** $p(x)$, which estimates the probability of a data point $x$ being normal or an anomaly.

What we can do is, for a given data point $x$, we can compute $p(x)$, and if $p(x) < \epsilon$, then we can flag it as an anomaly.

<img src="images/aircraft-engine-density-estimation.png" alt="Aircraft Engine Density Estimation" width="500"/>

Here, in the above image, you can see 3 different regions:
- The innermost region is the **normal region** where most of the data points are clustered, which has high probability density.
- The middle region is the **transition region** where the probability density is moderate.
- The outermost region is the **anomalous region** where the probability density is very low, that region we can set at a threshold $\epsilon$.

---

### Gaussian Distribution (Normal Distribution)

The **Gaussian Distribution** is also known as the **Normal Distribution**.

Gaussian Distribution is a continuous probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve, so it is also known as the **bell-shaped curve**.

#### Properties of Gaussian Distribution
- The Gaussian Distribution is defined by two parameters:
    - mean $\mu$ 
    - Variance $\sigma^2$
- The mean $\mu$ determines the location of the center of the peak of the bell curve.
- The variance $\sigma^2$ determines the width of the bell curve. The standard deviation $\sigma$ is the square root of the variance.
- The **Probability Density Function (PDF)** of the Gaussian Distribution is given by:
  $$ p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$

<img src="images/gaussian-distribution.png" alt="Gaussian Distribution" width="500"/>

#### Normal Distribution Variants
In the below image, you can see different variants of the Gaussian Distribution based on the mean and variance.

Some observations:
- The **mean** $\mu$ determines the location of the peak of the bell curve.
- The **variance** $\sigma^2$ (square of **standard deviation**) determines the width of the bell curve.
- Less variance means the data points are closer to the mean, and more variance means the data points are spread out.

<img src="images/normal_distribution-variants.png" alt="Gaussian Distribution PDF" height="600"/>

> #### Probability Density Function (PDF)
> The **Probability Density Function (PDF)** of a continuous random variable gives the relative likelihood of the random variable to take on a given value.
> - The PDF is a non-negative function, and the area under the PDF curve over an interval gives the probability of the random variable falling within that interval.
> - The PDF is a function of the random variable $x$ and is denoted as $p(x)$.
> 
> Sum of the probabilities of all possible outcomes of a random variable is equal to $1$.


#### Parameters Estimation

Given a dataset $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$, we can estimate the parameters $\mu$ and $\sigma^2$ of the Gaussian Distribution.

<img src="images/gaussian_distribution_sample.png" alt="Gaussian Distribution Parameters Estimation" width="500"/>

The parameters $\mu$ and $\sigma^2$ can be estimated as:
- The mean $\mu$ is the average of the data points:
  $$ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} $$

- The variance $\sigma^2$ is the average squared difference from the mean:
 $$ \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2 $$

- The standard deviation $\sigma$ is the square root of the variance:
  $$ \sigma = \sqrt{\sigma^2} $$

---

### Anomaly Detection Algorithm

