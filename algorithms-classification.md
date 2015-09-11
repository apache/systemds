---
layout: global
title: SystemML Algorithms Reference - Classification
displayTitle: <a href="algorithms-reference.html">SystemML Algorithms Reference</a>
---

     
# 2. Classification


## 2.1. Multinomial Logistic Regression

### Description

The `MultiLogReg.dml` script performs both binomial and multinomial
logistic regression. The script is given a dataset $(X, Y)$ where matrix
$X$ has $m$ columns and matrix $Y$ has one column; both $X$ and $Y$ have
$n$ rows. The rows of $X$ and $Y$ are viewed as a collection of records:
$(X, Y) = (x_i, y_i)_{i=1}^n$ where $x_i$ is a numerical vector of
explanatory (feature) variables and $y_i$ is a categorical response
variable. Each row $x_i$ in $X$ has size $\dim x_i = m$, while its corresponding $y_i$
is an integer that represents the observed response value for
record $i$.

The goal of logistic regression is to learn a linear model over the
feature vector $x_i$ that can be used to predict how likely each
categorical label is expected to be observed as the actual $y_i$. Note
that logistic regression predicts more than a label: it predicts the
probability for every possible label. The binomial case allows only two
possible labels, the multinomial case has no such restriction.

Just as linear regression estimates the mean value $\mu_i$ of a
numerical response variable, logistic regression does the same for
category label probabilities. In linear regression, the mean of $y_i$ is
estimated as a linear combination of the features:
$$\mu_i = \beta_0 + \beta_1 x_{i,1} + \ldots + \beta_m x_{i,m} = \beta_0 + x_i\beta_{1:m}$$.
In logistic regression, the label probability has to lie between 0
and 1, so a link function is applied to connect it to
$\beta_0 + x_i\beta_{1:m}$. If there are just two possible category
labels, for example 0 and 1, the logistic link looks as follows:

$$Prob[y_i\,{=}\,1\mid x_i; \beta] \,=\, 
\frac{e^{\,\beta_0 + x_i\beta_{1:m}}}{1 + e^{\,\beta_0 + x_i\beta_{1:m}}};
\quad
Prob[y_i\,{=}\,0\mid x_i; \beta] \,=\, 
\frac{1}{1 + e^{\,\beta_0 + x_i\beta_{1:m}}}$$

Here category label 0
serves as the *baseline*, and function $$\exp(\beta_0 + x_i\beta_{1:m})$$
shows how likely we expect to see “$y_i = 1$” in comparison to the
baseline. Like in a loaded coin, the predicted odds of seeing 1 versus 0
are $$\exp(\beta_0 + x_i\beta_{1:m})$$ to 1, with each feature $$x_{i,j}$$
multiplying its own factor $\exp(\beta_j x_{i,j})$ to the odds. Given a
large collection of pairs $(x_i, y_i)$, $i=1\ldots n$, logistic
regression seeks to find the $\beta_j$’s that maximize the product of
probabilities $Prob[y_i\mid x_i; \beta]$
for actually observed $y_i$-labels (assuming no
regularization).

Multinomial logistic regression [[Agresti2002]](algorithms-bibliography.html)
extends this link to
$k \geq 3$ possible categories. Again we identify one category as the
baseline, for example the $k$-th category. Instead of a coin, here we
have a loaded multisided die, one side per category. Each non-baseline
category $l = 1\ldots k\,{-}\,1$ has its own vector
$$(\beta_{0,l}, \beta_{1,l}, \ldots, \beta_{m,l})$$ of regression
parameters with the intercept, making up a matrix $B$ of size
$(m\,{+}\,1)\times(k\,{-}\,1)$. The predicted odds of seeing
non-baseline category $l$ versus the baseline $k$ are
$$\exp\big(\beta_{0,l} + \sum\nolimits_{j=1}^m x_{i,j}\beta_{j,l}\big)$$
to 1, and the predicted probabilities are: 

$$
\begin{equation}
l < k: Prob [y_i {=} l \mid x_i; B] \,\,\,{=}\,\,\,
\frac{\exp\big(\beta_{0,l} + \sum\nolimits_{j=1}^m x_{i,j}\beta_{j,l}\big)}{1 \,+\, \sum_{l'=1}^{k-1}\exp\big(\beta_{0,l'} + \sum\nolimits_{j=1}^m x_{i,j}\beta_{j,l'}\big)};
\end{equation}
$$

$$
\begin{equation}
Prob [y_i {=} k \mid x_i; B] \,\,\,{=}\,\,\,
\frac{1}{1 \,+\, \sum_{l'=1}^{k-1}\exp\big(\beta_{0,l'} + \sum\nolimits_{j=1}^m x_{i,j}\beta_{j,l'}\big)}.
\end{equation}
$$

The goal of the regression
is to estimate the parameter matrix $B$ from the provided dataset
$(X, Y) = (x_i, y_i)_{i=1}^n$ by maximizing the product of over the
observed labels $y_i$. Taking its logarithm, negating, and adding a
regularization term gives us a minimization objective:

$$
\begin{equation}
f(B; X, Y) \,\,=\,\,
-\sum_{i=1}^n \,\log Prob[y_i\mid x_i; B] \,+\,
\frac{\lambda}{2} \sum_{j=1}^m \sum_{l=1}^{k-1} |\beta_{j,l}|^2
\,\,\to\,\,\min
\end{equation}
$$

The optional regularization term is added to
mitigate overfitting and degeneracy in the data; to reduce bias, the
intercepts $$\beta_{0,l}$$ are not regularized. Once the $\beta_{j,l}$’s
are accurately estimated, we can make predictions about the category
label $y$ for a new feature vector $x$ using
Eqs. (1) and (2).


### Usage

    hadoop jar SystemML.jar -f MultiLogReg.dml
                            -nvargs X=file
                                    Y=file
                                    B=file
                                    Log=file
                                    icpt=int
                                    reg=double
                                    tol=double
                                    moi=int
                                    mii=int
                                    fmt=format


### Arguments

**X**: Location (on HDFS) to read the input matrix of feature vectors; each row
constitutes one feature vector.

**Y**: Location to read the input one-column matrix of category labels that
correspond to feature vectors in X. Note the following:\
– Each non-baseline category label must be a positive integer.\
– If all labels are positive, the largest represents the baseline
category.\
– If non-positive labels such as $-1$ or $0$ are present, then they
represent the (same) baseline category and are converted to label
$\max(\texttt{Y})\,{+}\,1$.

**B**: Location to store the matrix of estimated regression parameters (the
$$\beta_{j, l}$$’s), with the intercept parameters $\beta_{0, l}$ at
position B\[$m\,{+}\,1$, $l$\] if available.
The size of B is $(m\,{+}\,1)\times (k\,{-}\,1)$ with the
intercepts or $m \times (k\,{-}\,1)$ without the intercepts, one column
per non-baseline category and one row per feature.

**Log**: (default: " ") Location to store iteration-specific variables for monitoring
and debugging purposes, see 
[**Table 5**](algorithms-classification.html#table5)
for details.

**icpt**: (default: 0) Intercept and shifting/rescaling of the features in $X$:\
0 = no intercept (hence no $\beta_0$), no
shifting/rescaling of the features;\
1 = add intercept, but do not shift/rescale the features
in $X$;\
2 = add intercept, shift/rescale the features in $X$ to
mean 0, variance 1

**reg**: (default: 0.0) L2-regularization parameter (lambda)

**tol**: (default: 0.000001) Tolerance (epsilon) used in the convergence criterion

**moi**: (default: 100) Maximum number of outer (Fisher scoring) iterations

**mii**: (default: 0) Maximum number of inner (conjugate gradient) iterations, or 0
if no maximum limit provided

**fmt**: (default: `"text"`) Matrix file output format, such as text,
mm, or csv; see read/write functions in
SystemML Language Reference for details.


### Examples

    hadoop jar SystemML.jar -f MultiLogReg.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    fmt=csv
                                    icpt=2
                                    reg=1.0
                                    tol=0.0001
                                    moi=100
                                    mii=10
                                    Log=/user/ml/log.csv


* * *

<a name="table5" />
**Table 5**: The `Log` file for multinomial logistic regression
contains the following iteration variables in CSV format, each line
containing triple (Name, Iteration\#, Value) with Iteration\# being 0
for initial values.
  

| Name                | Meaning          |
| ------------------- | -------------------------- |
| LINEAR\_TERM\_MIN   | The minimum value of $X$ %*% $B$, used to check for overflows |
| LINEAR\_TERM\_MAX   | The maximum value of $X$ %*% $B$, used to check for overflows |
| NUM\_CG\_ITERS      | Number of inner (Conj. Gradient) iterations in this outer iteration |
| IS\_TRUST\_REACHED  | $1 = {}$trust region boundary was reached, $0 = {}$otherwise |
| POINT\_STEP\_NORM   | L2-norm of iteration step from old point (matrix $B$) to new point |
| OBJECTIVE           | The loss function we minimize (negative regularized log-likelihood) |
| OBJ\_DROP\_REAL     | Reduction in the objective during this iteration, actual value |
| OBJ\_DROP\_PRED     | Reduction in the objective predicted by a quadratic approximation |
| OBJ\_DROP\_RATIO    | Actual-to-predicted reduction ratio, used to update the trust region |
| IS\_POINT\_UPDATED  | $1 = {}$new point accepted; $0 = {}$new point rejected, old point restored |
| GRADIENT\_NORM      | L2-norm of the loss function gradient (omitted if point is rejected) |
| RUST\_DELTA         | Updated trust region size, the “delta” |


* * *

### Details

We estimate the logistic regression parameters via L2-regularized
negative log-likelihood minimization (3). The
optimization method used in the script closely follows the trust region
Newton method for logistic regression described in [[Lin2008]](algorithms-bibliography.html).
For convenience, let us make some changes in notation:

  * Convert the input vector of observed category labels into an indicator
matrix $Y$ of size $n \times k$ such that $$Y_{i, l} = 1$$ if the $i$-th
category label is $l$ and $Y_{i, l} = 0$ otherwise.
  * Append an extra column of all ones, i.e. $(1, 1, \ldots, 1)^T$, as the
$m\,{+}\,1$-st column to the feature matrix $X$ to represent the
intercept.
  * Append an all-zero column as the $k$-th column to $B$, the matrix of
regression parameters, to represent the baseline category.
  * Convert the regularization constant $\lambda$ into matrix $\Lambda$ of
the same size as $B$, placing 0’s into the $m\,{+}\,1$-st row to disable
intercept regularization, and placing $\lambda$’s everywhere else.

Now the ($n\,{\times}\,k$)-matrix of predicted probabilities given by
(1) and (2) and the
objective function $f$ in (3) have the matrix form

$$\begin{aligned}
P \,\,&=\,\, \exp(XB) \,\,/\,\, \big(\exp(XB)\,1_{k\times k}\big)\\
f \,\,&=\,\, - \,\,{\textstyle\sum} \,\,Y \cdot (X B)\, + \,
{\textstyle\sum}\,\log\big(\exp(XB)\,1_{k\times 1}\big) \,+ \,
(1/2)\,\, {\textstyle\sum} \,\,\Lambda \cdot B \cdot B\end{aligned}$$

where operations $\cdot\,$, $/$, $\exp$, and $\log$ are applied
cellwise, and $\textstyle\sum$ denotes the sum of all cells in a matrix.
The gradient of $f$ with respect to $B$ can be represented as a matrix
too:

$$\nabla f \,\,=\,\, X^T (P - Y) \,+\, \Lambda \cdot B$$

The Hessian $\mathcal{H}$ of $f$ is a tensor, but, fortunately, the
conjugate gradient inner loop of the trust region algorithm
in [[Lin2008]](algorithms-bibliography.html)
does not need to instantiate it. We only need to
multiply $\mathcal{H}$ by ordinary matrices of the same size as $B$ and
$\nabla f$, and this can be done in matrix form:

$$\mathcal{H}V \,\,=\,\, X^T \big( Q \,-\, P \cdot (Q\,1_{k\times k}) \big) \,+\,
\Lambda \cdot V, \,\,\,\,\textrm{where}\,\,\,\,Q \,=\, P \cdot (XV)$$

At each Newton iteration (the *outer* iteration) the minimization algorithm
approximates the difference
$\varDelta f(S; B) = f(B + S; X, Y) \,-\, f(B; X, Y)$ attained in the
objective function after a step $B \mapsto B\,{+}\,S$ by a second-degree
formula

$$\varDelta f(S; B) \,\,\,\approx\,\,\, (1/2)\,\,{\textstyle\sum}\,\,S \cdot \mathcal{H}S
 \,+\, {\textstyle\sum}\,\,S\cdot \nabla f$$
 
This approximation is then
minimized by trust-region conjugate gradient iterations (the *inner*
iterations) subject to the constraint
$\|S\|_2 \leq \delta$
. The trust
region size $\delta$ is initialized as
$0.5\sqrt{m}\,/ \max_i \|x_i\|_2$
and updated as described
in [[Lin2008]](algorithms-bibliography.html).
Users can specify the maximum number of the outer
and the inner iterations with input parameters moi and
mii, respectively. The iterative minimizer terminates
successfully if
$$\|\nabla f\|_2 < \varepsilon \|\nabla f_{B=0} \|_2$$
, where ${\varepsilon}> 0$ is a tolerance supplied by the user via input
parameter tol.


### Returns

The estimated regression parameters (the
$$\hat{\beta}_{j, l}$$)
are
populated into a matrix and written to an HDFS file whose path/name was
provided as the “B” input argument. Only the non-baseline
categories ($1\leq l \leq k\,{-}\,1$) have their 
$$\hat{\beta}_{j, l}$$
in the output; to add the baseline category, just append a column of zeros.
If icpt=0 in the input command line, no intercepts are used
and B has size 
$m\times (k\,{-}\,1)$; otherwise
B has size 
$(m\,{+}\,1)\times (k\,{-}\,1)$
and the
intercepts are in the 
$m\,{+}\,1$-st row. If icpt=2, then
initially the feature columns in $X$ are shifted to mean${} = 0$ and
rescaled to variance${} = 1$. After the iterations converge, the
$\hat{\beta}_{j, l}$’s are rescaled and shifted to work with the
original features.


* * *

## 2.2 Support Vector Machines

### 2.2.1 Binary-Class Support Vector Machines

#### Description

Support Vector Machines are used to model the relationship between a
categorical dependent variable y and one or more explanatory variables
denoted X. This implementation learns (and predicts with) a binary class
support vector machine (y with domain size 2).


#### Usage

    hadoop jar SystemML.jar -f l2-svm.dml
                            -nvargs X=file
                                    Y=file
                                    icpt=int
                                    tol=double
                                    reg=double
                                    maxiter=int
                                    model=file
                                    Log=file
                                    fmt=format

    hadoop jar SystemML.jar -f l2-svm-predict.dml
                            -nvargs X=file
                                    Y=file
                                    icpt=int
                                    model=file
                                    scores=file
                                    accuracy=file
                                    confusion=file
                                    fmt=format


#### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors; each
row constitutes one feature vector.

**Y**: Location to read the one-column matrix of (categorical) labels
that correspond to feature vectors in X. Binary class labels can be
expressed in one of two choices: $\pm 1$ or $1/2$. Note that, this
argument is optional for prediction.

**icpt**: (default: 0) If set to 1 then a constant bias
column is added to X.

**tol**: (default: 0.001) Procedure terminates early if the
reduction in objective function value is less than tolerance times
the initial objective function value.

**reg**: (default: 1) Regularization constant. See details
to find out where lambda appears in the objective function. If one
were interested in drawing an analogy with the C parameter in C-SVM,
then C = 2/lambda. Usually, cross validation is employed to
determine the optimum value of lambda.

**maxiter**: (default: 100) The maximum number
of iterations.

**model**: Location (on HDFS) that contains the learnt weights.

**Log**: Location (on HDFS) to collect various metrics (e.g., objective
function value etc.) that depict progress across iterations
while training.

**fmt**: (default: `"text"`) Specifies the output format.
Choice of comma-separated values (csv) or as a sparse-matrix (text).

**scores**: Location (on HDFS) to store scores for a held-out test set.
Note that, this is an optional argument.

**accuracy**: Location (on HDFS) to store the accuracy computed on a
held-out test set. Note that, this is an optional argument.

**confusion**: Location (on HDFS) to store the confusion matrix computed
using a held-out test set. Note that, this is an optional argument.


#### Examples

    hadoop jar SystemML.jar -f l2-svm.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/y.mtx
                                    icpt=0
                                    tol=0.001
                                    fmt=csv
                                    reg=1.0
                                    maxiter=100
                                    model=/user/ml/weights.csv
                                    Log=/user/ml/Log.csv

    hadoop jar SystemML.jar -f l2-svm-predict.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/y.mtx
                                    icpt=0
                                    fmt=csv
                                    model=/user/ml/weights.csv
                                    scores=/user/ml/scores.csv
                                    accuracy=/user/ml/accuracy.csv
                                    confusion=/user/ml/confusion.csv


#### Details

Support vector machines learn a classification function by solving the
following optimization problem ($L_2$-SVM):

$$\begin{aligned}
&\textrm{argmin}_w& \frac{\lambda}{2} ||w||_2^2 + \sum_i \xi_i^2\\
&\textrm{subject to:}& y_i w^{\top} x_i \geq 1 - \xi_i ~ \forall i\end{aligned}$$

where $x_i$ is an example from the training set with its label given by
$y_i$, $w$ is the vector of parameters and $\lambda$ is the
regularization constant specified by the user.

To account for the missing bias term, one may augment the data with a
column of constants which is achieved by setting the intercept argument to 1
[[Hsieh2008]](algorithms-bibliography.html).

This implementation optimizes the primal directly
[[Chapelle2007]](algorithms-bibliography.html). It
uses nonlinear conjugate gradient descent to minimize the objective
function coupled with choosing step-sizes by performing one-dimensional
Newton minimization in the direction of the gradient.


#### Returns

The learnt weights produced by `l2-svm.dml` are populated into a single
column matrix and written to file on HDFS (see model in section
Arguments). The number of rows in this matrix is ncol(X) if intercept
was set to 0 during invocation and ncol(X) + 1 otherwise. The bias term,
if used, is placed in the last row. Depending on what arguments are
provided during invocation, `l2-svm-predict.dml` may compute one or more
of scores, accuracy and confusion matrix in the output format
specified.


* * *


### 2.2.2 Multi-Class Support Vector Machines

#### Description

Support Vector Machines are used to model the relationship between a
categorical dependent variable y and one or more explanatory variables
denoted X. This implementation supports dependent variables that have
domain size greater or equal to 2 and hence is not restricted to binary
class labels.


#### Usage

    hadoop jar SystemML.jar -f m-svm.dml
                            -nvargs X=file
                                    Y=file
                                    icpt=int
                                    tol=double
                                    reg=double
                                    maxiter=int
                                    model=file
                                    Log=file
                                    fmt=format

    hadoop jar SystemML.jar -f m-svm-predict.dml
                            -nvargs X=file
                                    Y=file
                                    icpt=int
                                    model=file
                                    scores=file
                                    accuracy=file
                                    confusion=file
                                    fmt=format


#### Arguments

**X**: Location (on HDFS) containing the explanatory variables in
    a matrix. Each row constitutes an example.

**Y**: Location (on HDFS) containing a 1-column matrix specifying the
    categorical dependent variable (label). Labels are assumed to be
    contiguously numbered from 1 $\ldots$ \#classes. Note that, this
    argument is optional for prediction.

**icpt**: (default: 0) If set to 1 then a constant bias
    column is added to X.

**tol**: (default: 0.001) Procedure terminates early if the
    reduction in objective function value is less than tolerance times
    the initial objective function value.

**reg**: (default: 1) Regularization constant. See details
    to find out where lambda appears in the objective function. If one
    were interested in drawing an analogy with C-SVM, then C = 2/lambda.
    Usually, cross validation is employed to determine the optimum value
    of lambda.

**maxiter**: (default: 100) The maximum number
    of iterations.

**model**: Location (on HDFS) that contains the learnt weights.

**Log**: Location (on HDFS) to collect various metrics (e.g., objective
    function value etc.) that depict progress across iterations
    while training.

**fmt**: (default: `"text"`) Specifies the output format.
    Choice of comma-separated values (csv) or as a sparse-matrix (text).

**scores**: Location (on HDFS) to store scores for a held-out test set.
    Note that, this is an optional argument.

**accuracy**: Location (on HDFS) to store the accuracy computed on a
    held-out test set. Note that, this is an optional argument.

**confusion**: Location (on HDFS) to store the confusion matrix computed
    using a held-out test set. Note that, this is an optional argument.


#### Examples

    hadoop jar SystemML.jar -f m-svm.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/y.mtx 
                                    icpt=0
                                    tol=0.001
                                    reg=1.0 
                                    maxiter=100 
                                    fmt=csv 
                                    model=/user/ml/weights.csv
                                    Log=/user/ml/Log.csv

    hadoop jar SystemML.jar -f m-svm-predict.dml 
                            -nvargs X=/user/ml/X.mtx 
                                    Y=/user/ml/y.mtx 
                                    icpt=0 
                                    fmt=csv
                                    model=/user/ml/weights.csv
                                    scores=/user/ml/scores.csv
                                    accuracy=/user/ml/accuracy.csv
                                    confusion=/user/ml/confusion.csv


#### Details

Support vector machines learn a classification function by solving the
following optimization problem ($L_2$-SVM):

$$\begin{aligned}
&\textrm{argmin}_w& \frac{\lambda}{2} ||w||_2^2 + \sum_i \xi_i^2\\
&\textrm{subject to:}& y_i w^{\top} x_i \geq 1 - \xi_i ~ \forall i\end{aligned}$$

where $x_i$ is an example from the training set with its label given by
$y_i$, $w$ is the vector of parameters and $\lambda$ is the
regularization constant specified by the user.

To extend the above formulation (binary class SVM) to the multiclass
setting, one standard approach is to learn one binary class SVM per
class that separates data belonging to that class from the rest of the
training data (one-against-the-rest SVM, see 
[[Scholkopf1995]](algorithms-bibliography.html)).

To account for the missing bias term, one may augment the data with a
column of constants which is achieved by setting intercept argument to 1
[[Hsieh2008]](algorithms-bibliography.html).

This implementation optimizes the primal directly
[[Chapelle2007]](algorithms-bibliography.html). It
uses nonlinear conjugate gradient descent to minimize the objective
function coupled with choosing step-sizes by performing one-dimensional
Newton minimization in the direction of the gradient.


#### Returns

The learnt weights produced by `m-svm.dml` are populated into a matrix
that has as many columns as there are classes in the training data, and
written to file provided on HDFS (see model in section Arguments). The
number of rows in this matrix is ncol(X) if intercept was set to 0
during invocation and ncol(X) + 1 otherwise. The bias terms, if used,
are placed in the last row. Depending on what arguments are provided
during invocation, `m-svm-predict.dml` may compute one or more of scores,
accuracy and confusion matrix in the output format specified.


* * *

## 2.3 Naive Bayes

### Description

Naive Bayes is very simple generative model used for classifying data.
This implementation learns a multinomial naive Bayes classifier which is
applicable when all features are counts of categorical values.


#### Usage

    hadoop jar SystemML.jar -f naive-bayes.dml
                            -nvargs X=file
                                    Y=file
                                    laplace=double
                                    prior=file
                                    conditionals=file
                                    accuracy=file
                                    fmt=format

    hadoop jar SystemML.jar -f naive-bayes-predict.dml
                            -nvargs X=file
                                    Y=file
                                    prior=file
                                    conditionals=file
                                    fmt=format
                                    accuracy=file
                                    confusion=file
                                    probabilities=file


### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors; each
    row constitutes one feature vector.

**Y**: Location (on HDFS) to read the one-column matrix of (categorical)
    labels that correspond to feature vectors in X. Classes are assumed
    to be contiguously labeled beginning from 1. Note that, this
    argument is optional for prediction.

**laplace**: (default: 1) Laplace smoothing specified by
    the user to avoid creation of 0 probabilities.

**prior**: Location (on HDFS) that contains the class
    prior probabilites.

**conditionals**: Location (on HDFS) that contains the class conditional
    feature distributions.

**fmt** (default: `"text"`): Specifies the output format.
    Choice of comma-separated values (csv) or as a sparse-matrix (text).

**probabilities**: Location (on HDFS) to store class membership
    probabilities for a held-out test set. Note that, this is an
    optional argument.

**accuracy**: Location (on HDFS) to store the training accuracy during
    learning and testing accuracy from a held-out test set
    during prediction. Note that, this is an optional argument
    for prediction.

**confusion**: Location (on HDFS) to store the confusion matrix computed
    using a held-out test set. Note that, this is an optional argument.


### Examples

    hadoop jar SystemML.jar -f naive-bayes.dml
                            -nvargs X=/user/ml/X.mtx 
                                    Y=/user/ml/y.mtx 
                                    laplace=1 fmt=csv
                                    prior=/user/ml/prior.csv
                                    conditionals=/user/ml/conditionals.csv
                                    accuracy=/user/ml/accuracy.csv

    hadoop jar SystemML.jar -f naive-bayes-predict.dml
                            -nvargs X=/user/ml/X.mtx 
                                    Y=/user/ml/y.mtx 
                                    prior=/user/ml/prior.csv
                                    conditionals=/user/ml/conditionals.csv
                                    fmt=csv
                                    accuracy=/user/ml/accuracy.csv
                                    probabilities=/user/ml/probabilities.csv
                                    confusion=/user/ml/confusion.csv


### Details

Naive Bayes is a very simple generative classification model. It posits
that given the class label, features can be generated independently of
each other. More precisely, the (multinomial) naive Bayes model uses the
following equation to estimate the joint probability of a feature vector
$x$ belonging to class $y$:

$$\text{Prob}(y, x) = \pi_y \prod_{i \in x} \theta_{iy}^{n(i,x)}$$ 

where $\pi_y$ denotes the prior probability of class $y$, $i$ denotes a
feature present in $x$ with $n(i,x)$ denoting its count and
$\theta_{iy}$ denotes the class conditional probability of feature $i$
in class $y$. The usual constraints hold on $\pi$ and $\theta$:

$$\begin{aligned}
&& \pi_y \geq 0, ~ \sum_{y \in \mathcal{C}} \pi_y = 1\\
\forall y \in \mathcal{C}: && \theta_{iy} \geq 0, ~ \sum_i \theta_{iy} = 1\end{aligned}$$

where $\mathcal{C}$ is the set of classes.

Given a fully labeled training dataset, it is possible to learn a naive
Bayes model using simple counting (group-by aggregates). To compute the
class conditional probabilities, it is usually advisable to avoid
setting $\theta_{iy}$ to 0. One way to achieve this is using additive
smoothing or Laplace smoothing. Some authors have argued that this
should in fact be add-one smoothing. This implementation uses add-one
smoothing by default but lets the user specify her/his own constant, if
required.

This implementation is sometimes referred to as *multinomial* naive
Bayes. Other flavours of naive Bayes are also popular.


### Returns

The learnt model produced by `naive-bayes.dml` is stored in two separate
files. The first file stores the class prior (a single-column matrix).
The second file stores the class conditional probabilities organized
into a matrix with as many rows as there are class labels and as many
columns as there are features. Depending on what arguments are provided
during invocation, `naive-bayes-predict.dml` may compute one or more of
probabilities, accuracy and confusion matrix in the output format
specified.

    