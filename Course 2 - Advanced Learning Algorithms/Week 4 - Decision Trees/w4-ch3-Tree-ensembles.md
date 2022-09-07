# Week 4: Decision Trees

## Ch 3: Tree Ensembles

### Using multiple decision trees

-   Suppose we are doing _Cat Classification_ problem,
-   We have these $10$ _training_ examples:

<img src="./images/old-example.jpg" alt="10 cat classification example" width="500px" style="padding:10px" >

-   And $3$ input features:
-   -   _Face Shape_
-   -   _Ear Shape_
-   -   _Whiskers_
-   And target variable $y$ _Cat_
-   Say, the best feature with less entropy is _Ear Shape_.
-   After splitting the data, it looks like this:

<img src="./images/cat-ear-shape.jpg" alt="ear shape tree" width="300px" style="padding:10px 50px">
