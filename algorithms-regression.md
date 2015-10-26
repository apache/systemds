---
layout: global
title: SystemML Algorithms Reference - Regression
displayTitle: <a href="algorithms-reference.html">SystemML Algorithms Reference</a>
---

# 4. Regression


## 4.1. Linear Regression

### Description

Linear Regression scripts are used to model the relationship between one
numerical response variable and one or more explanatory (feature)
variables. The scripts are given a dataset $(X, Y) = (x_i, y_i)_{i=1}^n$
where $x_i$ is a numerical vector of feature variables and $y_i$ is a
numerical response value for each training data record. The feature
vectors are provided as a matrix $X$ of size $n\,{\times}\,m$, where $n$
is the number of records and $m$ is the number of features. The observed
response values are provided as a 1-column matrix $Y$, with a numerical
value $y_i$ for each $x_i$ in the corresponding row of matrix $X$.

In linear regression, we predict the distribution of the response $y_i$
based on a fixed linear combination of the features in $x_i$. We assume
that there exist constant regression coefficients
$\beta_0, \beta_1, \ldots, \beta_m$ and a constant residual
variance $\sigma^2$ such that

$$
\begin{equation}
y_i \sim Normal(\mu_i, \sigma^2) \,\,\,\,\textrm{where}\,\,\,\,
\mu_i \,=\, \beta_0 + \beta_1 x_{i,1} + \ldots + \beta_m x_{i,m}
\end{equation}
$$ 

Distribution
$y_i \sim Normal(\mu_i, \sigma^2)$
models the “unexplained” residual noise and is assumed independent
across different records.

The goal is to estimate the regression coefficients and the residual
variance. Once they are accurately estimated, we can make predictions
about $y_i$ given $x_i$ in new records. We can also use the $\beta_j$’s
to analyze the influence of individual features on the response value,
and assess the quality of this model by comparing residual variance in
the response, left after prediction, with its total variance.

There are two scripts in our library, both doing the same estimation,
but using different computational methods. Depending on the size and the
sparsity of the feature matrix $X$, one or the other script may be more
efficient. The “direct solve” script `LinearRegDS.dml` is more
efficient when the number of features $m$ is relatively small
($m \sim 1000$ or less) and matrix $X$ is either tall or fairly dense
(has ${\gg}\:m^2$ nonzeros); otherwise, the “conjugate gradient” script
`LinearRegCG.dml` is more efficient. If $m > 50000$, use only
`LinearRegCG.dml`.


### Usage

**Linear Regression - Direct Solve**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f LinearRegDS.dml
                            -nvargs X=<file>
                                    Y=<file>
                                    B=<file>
                                    O=[file]
                                    icpt=[int]
                                    reg=[double]
                                    fmt=[format]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f LinearRegDS.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=<file>
                                         B=<file>
                                         O=[file]
                                         icpt=[int]
                                         reg=[double]
                                         fmt=[format]
</div>
</div>

**Linear Regression - Conjugate Gradient**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f LinearRegCG.dml
                            -nvargs X=<file>
                                    Y=<file>
                                    B=<file>
                                    O=[file]
                                    Log=[file]
                                    icpt=[int]
                                    reg=[double]
                                    tol=[double]
                                    maxi=[int]
                                    fmt=[format]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f LinearRegCG.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=<file>
                                         B=<file>
                                         O=[file]
                                         Log=[file]
                                         icpt=[int]
                                         reg=[double]
                                         tol=[double]
                                         maxi=[int]
                                         fmt=[format]
</div>
</div>


### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors, each row
constitutes one feature vector

**Y**: Location to read the 1-column matrix of response values

**B**: Location to store the estimated regression parameters (the $\beta_j$’s),
with the intercept parameter $\beta_0$ at position
B\[$m\,{+}\,1$, 1\] if available

**O**: (default: `" "`) Location to store the CSV-file of summary statistics defined
in [**Table 7**](algorithms-regression.html#table7), the default is to print it to the
standard output

**Log**: (default: `" "`, `LinearRegCG.dml` only) Location to store
iteration-specific variables for monitoring and debugging purposes, see
[**Table 8**](algorithms-regression.html#table8)
for details.

**icpt**: (default: `0`) Intercept presence and shifting/rescaling the features
in $X$:

  * 0 = no intercept (hence no $\beta_0$), no shifting or
rescaling of the features
  * 1 = add intercept, but do not shift/rescale the features
in $X$
  * 2 = add intercept, shift/rescale the features in $X$ to
mean 0, variance 1

**reg**: (default: `0.000001`) L2-regularization parameter $\lambda\geq 0$; set to nonzero for highly
dependent, sparse, or numerous ($m \gtrsim n/10$) features

**tol**: (default: `0.000001`, `LinearRegCG.dml` only) Tolerance $\varepsilon\geq 0$ used in the
convergence criterion: we terminate conjugate gradient iterations when
the $\beta$-residual reduces in L2-norm by this factor

**maxi**: (default: `0`, `LinearRegCG.dml` only) Maximum number of conjugate
gradient iterations, or `0` if no maximum limit provided

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.


### Examples

**Linear Regression - Direct Solve**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f LinearRegDS.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    fmt=csv
                                    O=/user/ml/stats.csv
                                    icpt=2
                                    reg=1.0
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f LinearRegDS.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         Y=/user/ml/Y.mtx
                                         B=/user/ml/B.mtx
                                         fmt=csv
                                         O=/user/ml/stats.csv
                                         icpt=2
                                         reg=1.0
</div>
</div>

**Linear Regression - Conjugate Gradient**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f LinearRegCG.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    fmt=csv
                                    O=/user/ml/stats.csv
                                    icpt=2
                                    reg=1.0
                                    tol=0.00000001
                                    maxi=100
                                    Log=/user/ml/log.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f LinearRegCG.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         Y=/user/ml/Y.mtx
                                         B=/user/ml/B.mtx
                                         fmt=csv
                                         O=/user/ml/stats.csv
                                         icpt=2
                                         reg=1.0
                                         tol=0.00000001
                                         maxi=100
                                         Log=/user/ml/log.csv
</div>
</div>


* * *

<a name="table7" />
**Table 7**: Besides $\beta$, linear regression scripts compute a few summary statistics
listed below.  The statistics are provided in CSV format, one comma-separated name-value
pair per each line.

| Name                  | Meaning |
| --------------------- | ------- |
| AVG\_TOT\_Y           | Average of the response value $Y$
| STDEV\_TOT\_Y         | Standard Deviation of the response value $Y$
| AVG\_RES\_Y           | Average of the residual $Y - \mathop{\mathrm{pred}}(Y \mid X)$, i.e. residual bias
| STDEV\_RES\_Y         | Standard Deviation of the residual $Y - \mathop{\mathrm{pred}}(Y \mid X)$
| DISPERSION            | GLM-style dispersion, i.e. residual sum of squares / \#deg. fr.
| PLAIN\_R2             | Plain $R^2$ of residual with bias included vs. total average
| ADJUSTED\_R2          | Adjusted $R^2$ of residual with bias included vs. total average
| PLAIN\_R2\_NOBIAS     | Plain $R^2$ of residual with bias subtracted vs. total average
| ADJUSTED\_R2\_NOBIAS  | Adjusted $R^2$ of residual with bias subtracted vs. total average
| PLAIN\_R2\_VS\_0      | * Plain $R^2$ of residual with bias included vs. zero constant
| ADJUSTED\_R2\_VS\_0   | * Adjusted $R^2$ of residual with bias included vs. zero constant

\* The last two statistics are only printed if there is no intercept (`icpt=0`)

* * *

<a name="table8" />
**Table 8**: The `Log` file for the `LinearRegCG.dml` script
contains the above iteration variables in CSV format, each line
containing a triple (Name, Iteration\#, Value) with Iteration\# being 0
for initial values.


| Name                  | Meaning |
| --------------------- | ------- |
| CG\_RESIDUAL\_NORM    | L2-norm of conjug. grad. residual, which is $A$ %\*% $\beta - t(X)$ %\*% $y$ where $A = t(X)$ %\*% $X + diag(\lambda)$, or a similar quantity
| CG\_RESIDUAL\_RATIO   | Ratio of current L2-norm of conjug. grad. residual over the initial

* * * 


### Details

To solve a linear regression problem over feature matrix $X$ and
response vector $Y$, we can find coefficients
$\beta_0, \beta_1, \ldots, \beta_m$ and $\sigma^2$ that maximize the
joint likelihood of all $y_i$ for $i=1\ldots n$, defined by the assumed
statistical model (1). Since the joint likelihood of the
independent
$y_i \sim Normal(\mu_i, \sigma^2)$
is proportional to the product of
$\exp\big({-}\,(y_i - \mu_i)^2 / (2\sigma^2)\big)$, we can take the
logarithm of this product, then multiply by $-2\sigma^2 < 0$ to obtain a
least squares problem: 

$$
\begin{equation}
\sum_{i=1}^n \, (y_i - \mu_i)^2 \,\,=\,\, 
\sum_{i=1}^n \Big(y_i - \beta_0 - \sum_{j=1}^m \beta_j x_{i,j}\Big)^2
\,\,\to\,\,\min
\end{equation}
$$ 

This may not be enough, however. The minimum may
sometimes be attained over infinitely many $\beta$-vectors, for example
if $X$ has an all-0 column, or has linearly dependent columns, or has
fewer rows than columns . Even if (2) has a unique
solution, other $\beta$-vectors may be just a little suboptimal[^1], yet
give significantly different predictions for new feature vectors. This
results in *overfitting*: prediction error for the training data ($X$
and $Y$) is much smaller than for the test data (new records).

Overfitting and degeneracy in the data is commonly mitigated by adding a
regularization penalty term to the least squares function:

$$
\begin{equation}
\sum_{i=1}^n \Big(y_i - \beta_0 - \sum_{j=1}^m \beta_j x_{i,j}\Big)^2
\,+\,\, \lambda \sum_{j=1}^m \beta_j^2
\,\,\to\,\,\min
\end{equation}
$$ 

The choice of $\lambda>0$, the regularization
constant, typically involves cross-validation where the dataset is
repeatedly split into a training part (to estimate the $\beta_j$’s) and
a test part (to evaluate prediction accuracy), with the goal of
maximizing the test accuracy. In our scripts, $\lambda$ is provided as
input parameter `reg`.

The solution to the least squares problem (3), through
taking the derivative and setting it to 0, has the matrix linear
equation form

$$
\begin{equation}
A\left[\textstyle\beta_{1:m}\atop\textstyle\beta_0\right] \,=\, \big[X,\,1\big]^T Y,\,\,\,
\textrm{where}\,\,\,
A \,=\, \big[X,\,1\big]^T \big[X,\,1\big]\,+\,\hspace{0.5pt} diag(\hspace{0.5pt}
\underbrace{\lambda,\ldots, \lambda}_{\scriptstyle m}, 0)
\end{equation}
$$ 

where $[X,\,1]$ is $X$ with an extra column of 1s
appended on the right, and the diagonal matrix of $\lambda$’s has a zero
to keep the intercept $\beta_0$ unregularized. If the intercept is
disabled by setting $icpt=0$, the equation is simply $X^T X \beta = X^T Y$.

We implemented two scripts for solving equation (4): one
is a “direct solver” that computes $A$ and then solves
$A\beta = [X,\,1]^T Y$ by calling an external package, the other
performs linear conjugate gradient (CG) iterations without ever
materializing $A$. The CG algorithm closely follows Algorithm 5.2 in
Chapter 5 of [[Nocedal2006]](algorithms-bibliography.html). Each step in the CG algorithm
computes a matrix-vector multiplication $q = Ap$ by first computing
$[X,\,1]\, p$ and then $[X,\,1]^T [X,\,1]\, p$. Usually the number of
such multiplications, one per CG iteration, is much smaller than $m$.
The user can put a hard bound on it with input
parameter `maxi`, or use the default maximum of $m+1$ (or $m$
if no intercept) by having `maxi=0`. The CG iterations
terminate when the L2-norm of vector $r = A\beta - [X,\,1]^T Y$
decreases from its initial value (for $\beta=0$) by the tolerance factor
specified in input parameter `tol`.

The CG algorithm is more efficient if computing
$[X,\,1]^T \big([X,\,1]\, p\big)$ is much faster than materializing $A$,
an $(m\,{+}\,1)\times(m\,{+}\,1)$ matrix. The Direct Solver (DS) is more
efficient if $X$ takes up a lot more memory than $A$ (i.e. $X$ has a lot
more nonzeros than $m^2$) and if $m^2$ is small enough for the external
solver ($m \lesssim 50000$). A more precise determination between CG
and DS is subject to further research.

In addition to the $\beta$-vector, the scripts estimate the residual
standard deviation $\sigma$ and the $R^2$, the ratio of “explained”
variance to the total variance of the response variable. These
statistics only make sense if the number of degrees of freedom
$n\,{-}\,m\,{-}\,1$ is positive and the regularization constant
$\lambda$ is negligible or zero. The formulas for $\sigma$ and
$R^2$ are:

$$R^2_{\textrm{plain}} = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}},\quad
\sigma \,=\, \sqrt{\frac{\mathrm{RSS}}{n - m - 1}},\quad
R^2_{\textrm{adj.}} = 1 - \frac{\sigma^2 (n-1)}{\mathrm{TSS}}$$ 

where

$$\mathrm{RSS} \,=\, \sum_{i=1}^n \Big(y_i - \hat{\mu}_i - 
\frac{1}{n} \sum_{i'=1}^n \,(y_{i'} - \hat{\mu}_{i'})\Big)^2; \quad
\mathrm{TSS} \,=\, \sum_{i=1}^n \Big(y_i - \frac{1}{n} \sum_{i'=1}^n y_{i'}\Big)^2$$

Here $\hat{\mu}_i$ are the predicted means for $y_i$ based on the
estimated regression coefficients and the feature vectors. They may be
biased when no intercept is present, hence the RSS formula subtracts the
bias.

Lastly, note that by choosing the input option `icpt=2` the
user can shift and rescale the columns of $X$ to have zero average and
the variance of 1. This is particularly important when using
regularization over highly disbalanced features, because regularization
tends to penalize small-variance columns (which need large $\beta_j$’s)
more than large-variance columns (with small $\beta_j$’s). At the end,
the estimated regression coefficients are shifted and rescaled to apply
to the original features.


### Returns

The estimated regression coefficients (the $\hat{\beta}_j$’s) are
populated into a matrix and written to an HDFS file whose path/name was
provided as the `B` input argument. What this matrix
contains, and its size, depends on the input argument `icpt`,
which specifies the user’s intercept and rescaling choice:

  * **icpt=0**: No intercept, matrix $B$ has size $m\,{\times}\,1$, with
$B[j, 1] = \hat{\beta}_j$ for each $j$ from 1 to $m = {}$ncol$(X)$.
  * **icpt=1**: There is intercept, but no shifting/rescaling of $X$; matrix $B$ has
size $(m\,{+}\,1) \times 1$, with $B[j, 1] = \hat{\beta}_j$ for $j$ from
1 to $m$, and $B[m\,{+}\,1, 1] = \hat{\beta}_0$, the estimated intercept
coefficient.
  * **icpt=2**: There is intercept, and the features in $X$ are shifted to mean$ = 0$
and rescaled to variance$ = 1$; then there are two versions of
the $\hat{\beta}_j$’s, one for the original features and another for the
shifted/rescaled features. Now matrix $B$ has size
$(m\,{+}\,1) \times 2$, with $B[\cdot, 1]$ for the original features and
$B[\cdot, 2]$ for the shifted/rescaled features, in the above format.
Note that $B[\cdot, 2]$ are iteratively estimated and $B[\cdot, 1]$ are
obtained from $B[\cdot, 2]$ by complementary shifting and rescaling.

The estimated summary statistics, including residual standard
deviation $\sigma$ and the $R^2$, are printed out or sent into a file
(if specified) in CSV format as defined in [**Table 7**](algorithms-regression.html#table7).
For conjugate gradient iterations, a log file with monitoring variables
can also be made available, see [**Table 8**](algorithms-regression.html#table8).


* * *

## 4.2. Stepwise Linear Regression

### Description

Our stepwise linear regression script selects a linear model based on
the Akaike information criterion (AIC): the model that gives rise to the
lowest AIC is computed.


### Usage

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f StepLinearRegDS.dml
                            -nvargs X=<file>
                                    Y=<file>
                                    B=<file>
                                    S=[file]
                                    O=[file]
                                    icpt=[int]
                                    thr=[double]
                                    fmt=[format]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f StepLinearRegDS.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=<file>
                                         B=<file>
                                         S=[file]
                                         O=[file]
                                         icpt=[int]
                                         thr=[double]
                                         fmt=[format]
</div>
</div>

### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors, each row
contains one feature vector.

**Y**: Location (on HDFS) to read the 1-column matrix of response values

**B**: Location (on HDFS) to store the estimated regression parameters (the
$\beta_j$’s), with the intercept parameter $\beta_0$ at position
B\[$m\,{+}\,1$, 1\] if available

**S**: (default: `" "`) Location (on HDFS) to store the selected feature-ids in the
order as computed by the algorithm; by default the selected feature-ids
are forwarded to the standard output.

**O**: (default: `" "`) Location (on HDFS) to store the CSV-file of summary
statistics defined in [**Table 7**](algorithms-regression.html#table7); by default the
summary statistics are forwarded to the standard output.

**icpt**: (default: `0`) Intercept presence and shifting/rescaling the features
in $X$:

  * 0 = no intercept (hence no $\beta_0$), no shifting or
rescaling of the features;
  * 1 = add intercept, but do not shift/rescale the features
in $X$;
  * 2 = add intercept, shift/rescale the features in $X$ to
mean 0, variance 1

**thr**: (default: `0.01`) Threshold to stop the algorithm: if the decrease in the value
of the AIC falls below `thr` no further features are being
checked and the algorithm stops.

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.


### Examples

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f StepLinearRegDS.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    S=/user/ml/selected.csv
                                    O=/user/ml/stats.csv
                                    icpt=2
                                    thr=0.05
                                    fmt=csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f StepLinearRegDS.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         Y=/user/ml/Y.mtx
                                         B=/user/ml/B.mtx
                                         S=/user/ml/selected.csv
                                         O=/user/ml/stats.csv
                                         icpt=2
                                         thr=0.05
                                         fmt=csv
</div>
</div>


### Details

Stepwise linear regression iteratively selects predictive variables in
an automated procedure. Currently, our implementation supports forward
selection: starting from an empty model (without any variable) the
algorithm examines the addition of each variable based on the AIC as a
model comparison criterion. The AIC is defined as

$$
\begin{equation}
AIC = -2 \log{L} + 2 edf,\label{eq:AIC}
\end{equation}
$$

where $L$ denotes the
likelihood of the fitted model and $edf$ is the equivalent degrees of
freedom, i.e., the number of estimated parameters. This procedure is
repeated until including no additional variable improves the model by a
certain threshold specified in the input parameter `thr`.

For fitting a model in each iteration we use the `direct solve` method
as in the script `LinearRegDS.dml` discussed in
[Linear Regression](algorithms-regression.html#linear-regression).


### Returns

Similar to the outputs from `LinearRegDS.dml` the stepwise
linear regression script computes the estimated regression coefficients
and stores them in matrix $B$ on HDFS. The format of matrix $B$ is
identical to the one produced by the scripts for linear regression (see
[Linear Regression](algorithms-regression.html#linear-regression)). Additionally, `StepLinearRegDS.dml`
outputs the variable indices (stored in the 1-column matrix $S$) in the
order they have been selected by the algorithm, i.e., $i$th entry in
matrix $S$ corresponds to the variable which improves the AIC the most
in $i$th iteration. If the model with the lowest AIC includes no
variables matrix $S$ will be empty (contains one 0). Moreover, the
estimated summary statistics as defined in [**Table 7**](algorithms-regression.html#table7)
are printed out or stored in a file (if requested). In the case where an
empty model achieves the best AIC these statistics will not be produced.


* * *

## 4.3. Generalized Linear Models

### Description

Generalized Linear Models 
[[Gill2000](algorithms-bibliography.html),
[McCullagh1989](algorithms-bibliography.html),
[Nelder1972](algorithms-bibliography.html)]
extend the methodology of linear
and logistic regression to a variety of distributions commonly assumed
as noise effects in the response variable. As before, we are given a
collection of records $(x_1, y_1)$, …, $(x_n, y_n)$ where $x_i$ is a
numerical vector of explanatory (feature) variables of size $\dim x_i = m$, and $y_i$
is the response (dependent) variable observed for this vector. GLMs
assume that some linear combination of the features in $x_i$ determines
the *mean* $\mu_i$ of $y_i$, while the observed $y_i$ is a random
outcome of a noise distribution
$Prob[y\mid \mu_i]\,$[^2]
with that mean $\mu_i$:

$$x_i \,\,\,\,\mapsto\,\,\,\, \eta_i = \beta_0 + \sum\nolimits_{j=1}^m \beta_j x_{i,j} 
\,\,\,\,\mapsto\,\,\,\, \mu_i \,\,\,\,\mapsto \,\,\,\, y_i \sim Prob[y\mid \mu_i]$$

In linear regression the response mean $\mu_i$ *equals* some linear
combination over $x_i$, denoted above by $\eta_i$. In logistic
regression with $$y\in\{0, 1\}$$ (Bernoulli) the mean of $y$ is the same
as $Prob[y=1]$
and equals $1/(1+e^{-\eta_i})$, the logistic function of $\eta_i$. In
GLM, $\mu_i$ and $\eta_i$ can be related via any given smooth monotone
function called the *link function*: $\eta_i = g(\mu_i)$. The unknown
linear combination parameters $\beta_j$ are assumed to be the same for
all records.

The goal of the regression is to estimate the parameters $\beta_j$ from
the observed data. Once the $\beta_j$’s are accurately estimated, we can
make predictions about $y$ for a new feature vector $x$. To do so,
compute $\eta$ from $x$ and use the inverted link function
$\mu = g^{-1}(\eta)$ to compute the mean $\mu$ of $y$; then use the
distribution $Prob[y\mid \mu]$
to make predictions about $y$. Both $g(\mu)$ and
$Prob[y\mid \mu]$
are user-provided. Our GLM script supports a standard set of
distributions and link functions, see below for details.


### Usage

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM.dml
                            -nvargs X=<file>
                                    Y=<file>
                                    B=<file>
                                    fmt=[format]
                                    O=[file]
                                    Log=[file]
                                    dfam=[int]
                                    vpow=[double]
                                    link=[int]
                                    lpow=[double]
                                    yneg=[double]
                                    icpt=[int]
                                    reg=[double]
                                    tol=[double]
                                    disp=[double]
                                    moi=[int]
                                    mii=[int]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=<file>
                                         B=<file>
                                         fmt=[format]
                                         O=[file]
                                         Log=[file]
                                         dfam=[int]
                                         vpow=[double]
                                         link=[int]
                                         lpow=[double]
                                         yneg=[double]
                                         icpt=[int]
                                         reg=[double]
                                         tol=[double]
                                         disp=[double]
                                         moi=[int]
                                         mii=[int]
</div>
</div>

### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors; each row
constitutes an example.

**Y**: Location to read the response matrix, which may have 1 or 2 columns

**B**: Location to store the estimated regression parameters (the $\beta_j$’s),
with the intercept parameter $\beta_0$ at position
B\[$m\,{+}\,1$, 1\] if available

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.

**O**: (default: `" "`) Location to write certain summary statistics described 
in [**Table 9**](algorithms-regression.html#table9), 
by default it is standard output.

**Log**: (default: `" "`) Location to store iteration-specific variables for monitoring
and debugging purposes, see [**Table 10**](algorithms-regression.html#table10) for details.

**dfam**: (default: `1`) Distribution family code to specify
$Prob[y\mid \mu]$,
see [**Table 11**](algorithms-regression.html#table11):

  * 1 = power distributions with $Var(y) = \mu^{\alpha}$
  * 2 = binomial or Bernoulli

**vpow**: (default: `0.0`) When `dfam=1`, this provides the $q$ in
$Var(y) = a\mu^q$, the power
dependence of the variance of $y$ on its mean. In particular, use:

  * 0.0 = Gaussian
  * 1.0 = Poisson
  * 2.0 = Gamma
  * 3.0 = inverse Gaussian

**link**: (default: `0`) Link function code to determine the link
function $\eta = g(\mu)$:

  * 0 = canonical link (depends on the distribution family),
see [**Table 11**](algorithms-regression.html#table11)
  * 1 = power functions
  * 2 = logit
  * 3 = probit
  * 4 = cloglog
  * 5 = cauchit

**lpow**: (default: `1.0`) When link=1, this provides the $s$ in
$\eta = \mu^s$, the power link function; `lpow=0.0` gives the
log link $\eta = \log\mu$. Common power links:

  * -2.0 = $1/\mu^2$ 
  * -1.0 = reciprocal
  * 0.0 = log 
  * 0.5 = sqrt 
  * 1.0 = identity

**yneg**: (default: `0.0`) When `dfam=2` and the response matrix $Y$ has
1 column, this specifies the $y$-value used for Bernoulli “No” label
(“Yes” is $1$):
0.0 when $y\in\\{0, 1\\}$; -1.0 when
$y\in\\{-1, 1\\}$

**icpt**: (default: `0`) Intercept and shifting/rescaling of the features in $X$:

  * 0 = no intercept (hence no $\beta_0$), no
shifting/rescaling of the features
  * 1 = add intercept, but do not shift/rescale the features
in $X$
  * 2 = add intercept, shift/rescale the features in $X$ to
mean 0, variance 1

**reg**: (default: `0.0`) L2-regularization parameter ($\lambda$)

**tol**: (default: `0.000001`) Tolerance ($\varepsilon$) used in the convergence criterion: we
terminate the outer iterations when the deviance changes by less than
this factor; see below for details

**disp**: (default: `0.0`) Dispersion parameter, or 0.0 to estimate it from
data

**moi**: (default: `200`) Maximum number of outer (Fisher scoring) iterations

**mii**: (default: `0`) Maximum number of inner (conjugate gradient) iterations, or 0
if no maximum limit provided


### Examples

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    fmt=csv
                                    dfam=2
                                    link=2
                                    yneg=-1.0
                                    icpt=2
                                    reg=0.01
                                    tol=0.00000001
                                    disp=1.0
                                    moi=100
                                    mii=10
                                    O=/user/ml/stats.csv
                                    Log=/user/ml/log.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         Y=/user/ml/Y.mtx
                                         B=/user/ml/B.mtx
                                         fmt=csv
                                         dfam=2
                                         link=2
                                         yneg=-1.0
                                         icpt=2
                                         reg=0.01
                                         tol=0.00000001
                                         disp=1.0
                                         moi=100
                                         mii=10
                                         O=/user/ml/stats.csv
                                         Log=/user/ml/log.csv
</div>
</div>

* * *

<a name="table9" />
**Table 9**: Besides $\beta$, GLM regression script computes a few summary
statistics listed below. They are provided in CSV format, one
comma-separated name-value pair per each line.

| Name                  | Meaning |
| --------------------- | ------- |
| TERMINATION\_CODE   | A positive integer indicating success/failure as follows: <br />1 = Converged successfully; 2 = Maximum \# of iterations reached; 3 = Input (X, Y) out of range; 4 = Distribution/link not supported
| BETA\_MIN           | Smallest beta value (regression coefficient), excluding the intercept
| BETA\_MIN\_INDEX    | Column index for the smallest beta value
| BETA\_MAX           | Largest beta value (regression coefficient), excluding the intercept
| BETA\_MAX\_INDEX    | Column index for the largest beta value
| INTERCEPT           | Intercept value, or NaN if there is no intercept (if `icpt=0`)
| DISPERSION          | Dispersion used to scale deviance, provided in disp input argument  or estimated (same as DISPERSION\_EST) if disp argument is $\leq 0$
| DISPERSION\_EST     | Dispersion estimated from the dataset
| DEVIANCE\_UNSCALED  | Deviance from the saturated model, assuming dispersion $= 1.0$
| DEVIANCE\_SCALED    | Deviance from the saturated model, scaled by DISPERSION value


* * *

<a name="table10" />
**Table 10**: The Log file for GLM regression contains the below
iteration variables in CSV format, each line containing a triple (Name,
Iteration\#, Value) with Iteration\# being 0 for initial values.

| Name                  | Meaning |
| --------------------- | ------- |
| NUM\_CG\_ITERS      | Number of inner (Conj. Gradient) iterations in this outer iteration
| IS\_TRUST\_REACHED  | 1 = trust region boundary was reached, 0 = otherwise
| POINT\_STEP\_NORM   | L2-norm of iteration step from old point ($\beta$-vector) to new point
| OBJECTIVE           | The loss function we minimize (negative partial log-likelihood)
| OBJ\_DROP\_REAL     | Reduction in the objective during this iteration, actual value
| OBJ\_DROP\_PRED     | Reduction in the objective predicted by a quadratic approximation
| OBJ\_DROP\_RATIO    | Actual-to-predicted reduction ratio, used to update the trust region
| GRADIENT\_NORM      | L2-norm of the loss function gradient (omitted if point is rejected)
| LINEAR\_TERM\_MIN   | The minimum value of $X$ %\*% $\beta$, used to check for overflows
| LINEAR\_TERM\_MAX   | The maximum value of $X$ %\*% $\beta$, used to check for overflows
| IS\_POINT\_UPDATED  | 1 = new point accepted; 0 = new point rejected, old point restored
| TRUST\_DELTA        | Updated trust region size, the “delta”


* * *

<a name="table11" />
**Table 11**: Common GLM distribution families and link functions. (Here “\*” stands
for “any value.”)

| dfam | vpow | link | lpow | Distribution<br />Family | Link<br /> Function | Canonical |
| :--: | :--: | :--: | :--: | :------------: | :--------: | :----: |
|   1  |  0.0 |   1  | -1.0 |    Gaussian    |   inverse  | 
|   1  |  0.0 |   1  |  0.0 |    Gaussian    |     log    | 
|   1  |  0.0 |   1  |  1.0 |    Gaussian    |  identity  |   Yes
|   1  |  1.0 |   1  |  0.0 |     Poisson    |     log    |   Yes
|   1  |  1.0 |   1  |  0.5 |     Poisson    |   sq.root  | 
|   1  |  1.0 |   1  |  1.0 |     Poisson    |  identity  | 
|   1  |  2.0 |   1  | -1.0 |      Gamma     |   inverse  |   Yes
|   1  |  2.0 |   1  |  0.0 |      Gamma     |     log    | 
|   1  |  2.0 |   1  |  1.0 |      Gamma     |  identity  | 
|   1  |  3.0 |   1  | -2.0 |  Inverse Gauss |  $1/\mu^2$ |   Yes
|   1  |  3.0 |   1  | -1.0 |  Inverse Gauss |   inverse  | 
|   1  |  3.0 |   1  |  0.0 |  Inverse Gauss |     log    | 
|   1  |  3.0 |   1  |  1.0 |  Inverse Gauss |  identity  | 
|   2  |   \* |   1  |  0.0 |    Binomial    |     log    | 
|   2  |   \* |   1  |  0.5 |    Binomial    |   sq.root  | 
|   2  |   \* |   2  |   \* |    Binomial    |    logit   |   Yes
|   2  |   \* |   3  |   \* |    Binomial    |   probit   | 
|   2  |   \* |   4  |   \* |    Binomial    |   cloglog  | 
|   2  |   \* |   5  |   \* |    Binomial    |   cauchit  | 


* * *

<a name="table12" />
**Table 12**: The supported non-power link functions for the Bernoulli and the
binomial distributions. Here $\mu$ is the Bernoulli mean.

| Name                  | Link Function |
| --------------------- | ------------- |
| Logit   | $\displaystyle \eta = 1 / \big(1 + e^{-\mu}\big)^{\mathstrut}$
| Probit  | $$\displaystyle \mu  = \frac{1}{\sqrt{2\pi}}\int\nolimits_{-\infty_{\mathstrut}}^{\,\eta\mathstrut} e^{-\frac{t^2}{2}} dt$$
| Cloglog | $\displaystyle \eta = \log \big(- \log(1 - \mu)\big)^{\mathstrut}$
| Cauchit | $\displaystyle \eta = \tan\pi(\mu - 1/2)$


* * *


### Details

In GLM, the noise distribution
$Prob[y\mid \mu]$
of the response variable $y$ given its mean $\mu$ is restricted to have
the *exponential family* form

$$
\begin{equation}
Y \sim\, Prob[y\mid \mu] \,=\, \exp\left(\frac{y\theta - b(\theta)}{a}
+ c(y, a)\right),\,\,\textrm{where}\,\,\,\mu = E(Y) = b'(\theta).
\end{equation}
$$ 

Changing the mean in such a distribution simply
multiplies all 
$Prob[y\mid \mu]$
by $e^{\,y\hspace{0.2pt}\theta/a}$ and rescales them so
that they again integrate to 1. Parameter $\theta$ is called
*canonical*, and the function $\theta = b'^{\,-1}(\mu)$ that relates it
to the mean is called the *canonical link*; constant $a$ is called
*dispersion* and rescales the variance of $y$. Many common distributions
can be put into this form, see [**Table 11**](algorithms-regression.html#table11). The canonical
parameter $\theta$ is often chosen to coincide with $\eta$, the linear
combination of the regression features; other choices for $\eta$ are
possible too.

Rather than specifying the canonical link, GLM distributions are
commonly defined by their variance
$Var(y)$ as the function of
the mean $\mu$. It can be shown from Eq. 5 that
$Var(y) = a\,b''(\theta) = a\,b''(b'^{\,-1}(\mu))$.
For example, for the Bernoulli distribution
$Var(y) = \mu(1-\mu)$, for the
Poisson distribution
$Var(y) = \mu$, and for the Gaussian distribution
$Var(y) = a\cdot 1 = \sigma^2$.
It turns out that for many common distributions
$Var(y) = a\mu^q$, a power
function. We support all distributions where
$Var(y) = a\mu^q$, as well as
the Bernoulli and the binomial distributions.

For distributions with
$Var(y) = a\mu^q$ the
canonical link is also a power function, namely
$\theta = \mu^{1-q}/(1-q)$, except for the Poisson ($q = 1$) whose
canonical link is $\theta = \log\mu$. We support all power link
functions in the form $\eta = \mu^s$, dropping any constant factor, with
$\eta = \log\mu$ for $s=0$. The binomial distribution has its own family
of link functions, which includes logit (the canonical link), probit,
cloglog, and cauchit (see [**Table 12**](algorithms-regression.html#table12)); we support
these only for the binomial and Bernoulli distributions. Links and
distributions are specified via four input parameters:
`dfam`, `vpow`, `link`, and
`lpow` (see [**Table 11**](algorithms-regression.html#table11)).

The observed response values are provided to the regression script as a
matrix $Y$ having 1 or 2 columns. If a power distribution family is
selected (`dfam=1`), matrix $Y$ must have 1 column that
provides $y_i$ for each $x_i$ in the corresponding row of matrix $X$.
When dfam=2 and $Y$ has 1 column, we assume the Bernoulli
distribution for $$y_i\in\{y_{\mathrm{neg}}, 1\}$$ with $y_{\mathrm{neg}}$
from the input parameter `yneg`. When `dfam=2` and
$Y$ has 2 columns, we assume the binomial distribution; for each row $i$
in $X$, cells $Y[i, 1]$ and $Y[i, 2]$ provide the positive and the
negative binomial counts respectively. Internally we convert the
1-column Bernoulli into the 2-column binomial with 0-versus-1 counts.

We estimate the regression parameters via L2-regularized negative
log-likelihood minimization:

$$f(\beta; X, Y) \,\,=\,\, -\sum\nolimits_{i=1}^n \big(y_i\theta_i - b(\theta_i)\big)
\,+\,(\lambda/2) \sum\nolimits_{j=1}^m \beta_j^2\,\,\to\,\,\min$$ 

where
$\theta_i$ and $b(\theta_i)$ are from (6); note that $a$ and
$c(y, a)$ are constant w.r.t. $\beta$ and can be ignored here. The
canonical parameter $\theta_i$ depends on both $\beta$ and $x_i$:

$$\theta_i \,\,=\,\, b'^{\,-1}(\mu_i) \,\,=\,\, b'^{\,-1}\big(g^{-1}(\eta_i)\big) \,\,=\,\,
\big(b'^{\,-1}\circ g^{-1}\big)\left(\beta_0 + \sum\nolimits_{j=1}^m \beta_j x_{i,j}\right)$$

The user-provided (via `reg`) regularization coefficient
$\lambda\geq 0$ can be used to mitigate overfitting and degeneracy in
the data. Note that the intercept is never regularized.

Our iterative minimizer for $f(\beta; X, Y)$ uses the Fisher scoring
approximation to the difference
$\varDelta f(z; \beta) = f(\beta + z; X, Y) \,-\, f(\beta; X, Y)$,
recomputed at each iteration: 

$$\begin{gathered}
\varDelta f(z; \beta) \,\,\,\approx\,\,\, 1/2 \cdot z^T A z \,+\, G^T z,
\,\,\,\,\textrm{where}\,\,\,\, A \,=\, X^T\!diag(w) X \,+\, \lambda I\\
\textrm{and}\,\,\,\,G \,=\, - X^T u \,+\, \lambda\beta,
\,\,\,\textrm{with $n\,{\times}\,1$ vectors $w$ and $u$ given by}\\
\forall\,i = 1\ldots n: \,\,\,\,
w_i = \big[v(\mu_i)\,g'(\mu_i)^2\big]^{-1}
\!\!\!\!\!\!,\,\,\,\,\,\,\,\,\,
u_i = (y_i - \mu_i)\big[v(\mu_i)\,g'(\mu_i)\big]^{-1}
\!\!\!\!\!\!.\,\,\,\,\end{gathered}$$ 

Here
$v(\mu_i)=Var(y_i)/a$, the
variance of $y_i$ as the function of the mean, and
$g'(\mu_i) = d \eta_i/d \mu_i$ is the link function derivative. The
Fisher scoring approximation is minimized by trust-region conjugate
gradient iterations (called the *inner* iterations, with the Fisher
scoring iterations as the *outer* iterations), which approximately solve
the following problem:

$$1/2 \cdot z^T A z \,+\, G^T z \,\,\to\,\,\min\,\,\,\,\textrm{subject to}\,\,\,\,
\|z\|_2 \leq \delta$$ 

The conjugate gradient algorithm closely follows
Algorithm 7.2 on page 171 of [[Nocedal2006]](algorithms-bibliography.html). The trust region
size $\delta$ is initialized as
$0.5\sqrt{m}\,/ \max\nolimits_i \|x_i\|_2$ and updated as described
in [[Nocedal2006]](algorithms-bibliography.html). 
The user can specify the maximum number of
the outer and the inner iterations with input parameters
`moi` and `mii`, respectively. The Fisher scoring
algorithm terminates successfully if
$2|\varDelta f(z; \beta)| < (D_1(\beta) + 0.1)\hspace{0.5pt}{\varepsilon}$
where ${\varepsilon}> 0$ is a tolerance supplied by the user via
`tol`, and $D_1(\beta)$ is the unit-dispersion deviance
estimated as

$$D_1(\beta) \,\,=\,\, 2 \cdot \big(Prob[Y \mid \!
\begin{smallmatrix}\textrm{saturated}\\\textrm{model}\end{smallmatrix}, a\,{=}\,1]
\,\,-\,\,Prob[Y \mid X, \beta, a\,{=}\,1]\,\big)$$

The deviance estimate is also produced as part of the output. Once the
Fisher scoring algorithm terminates, if requested by the user, we
estimate the dispersion $a$ from (6) using Pearson residuals

$$
\begin{equation}
\hat{a} \,\,=\,\, \frac{1}{n-m}\cdot \sum_{i=1}^n \frac{(y_i - \mu_i)^2}{v(\mu_i)}
\end{equation}
$$ 

and use it to adjust our deviance estimate:
$D_{\hat{a}}(\beta) = D_1(\beta)/\hat{a}$. If input argument
disp is 0.0 we estimate $\hat{a}$, otherwise
we use its value as $a$. Note that in (7) $m$ counts
the intercept ($m \leftarrow m+1$) if it is present.


### Returns

The estimated regression parameters (the $\hat{\beta}_j$’s) are
populated into a matrix and written to an HDFS file whose path/name was
provided as the `B` input argument. What this matrix
contains, and its size, depends on the input argument `icpt`,
which specifies the user’s intercept and rescaling choice:

  * **icpt=0**: No intercept, matrix $B$ has size $m\,{\times}\,1$, with
$B[j, 1] = \hat{\beta}_j$ for each $j$ from 1 to $m = {}$ncol$(X)$.
  * **icpt=1**: There is intercept, but no shifting/rescaling of $X$; matrix $B$ has
size $(m\,{+}\,1) \times 1$, with $B[j, 1] = \hat{\beta}_j$ for $j$ from
1 to $m$, and $B[m\,{+}\,1, 1] = \hat{\beta}_0$, the estimated intercept
coefficient.
  * **icpt=2**: There is intercept, and the features in $X$ are shifted to mean${} = 0$
and rescaled to variance${} = 1$; then there are two versions of
the $\hat{\beta}_j$’s, one for the original features and another for the
shifted/rescaled features. Now matrix $B$ has size
$(m\,{+}\,1) \times 2$, with $B[\cdot, 1]$ for the original features and
$B[\cdot, 2]$ for the shifted/rescaled features, in the above format.
Note that $B[\cdot, 2]$ are iteratively estimated and $B[\cdot, 1]$ are
obtained from $B[\cdot, 2]$ by complementary shifting and rescaling.

Our script also estimates the dispersion $\hat{a}$ (or takes it from the
user’s input) and the deviances $D_1(\hat{\beta})$ and
$D_{\hat{a}}(\hat{\beta})$, see [**Table 9**](algorithms-regression.html#table9) for details. A
log file with variables monitoring progress through the iterations can
also be made available, see [**Table 10**](algorithms-regression.html#table10).


### See Also

In case of binary classification problems, consider using L2-SVM or
binary logistic regression; for multiclass classification, use
multiclass SVM or multinomial logistic regression. For the special cases
of linear regression and logistic regression, it may be more efficient
to use the corresponding specialized scripts instead of GLM.


* * *

## 4.4. Stepwise Generalized Linear Regression

### Description

Our stepwise generalized linear regression script selects a model based
on the Akaike information criterion (AIC): the model that gives rise to
the lowest AIC is provided. Note that currently only the Bernoulli
distribution family is supported (see below for details).


### Usage

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f StepGLM.dml
                            -nvargs X=<file>
                                    Y=<file>
                                    B=<file>
                                    S=[file]
                                    O=[file]
                                    link=[int]
                                    yneg=[double]
                                    icpt=[int]
                                    tol=[double]
                                    disp=[double]
                                    moi=[int]
                                    mii=[int]
                                    thr=[double]
                                    fmt=[format]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f StepGLM.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=<file>
                                         B=<file>
                                         S=[file]
                                         O=[file]
                                         link=[int]
                                         yneg=[double]
                                         icpt=[int]
                                         tol=[double]
                                         disp=[double]
                                         moi=[int]
                                         mii=[int]
                                         thr=[double]
                                         fmt=[format]
</div>
</div>


### Arguments

**X**: Location (on HDFS) to read the matrix of feature vectors; each row is an
example.

**Y**: Location (on HDFS) to read the response matrix, which may have 1 or 2
columns

**B**: Location (on HDFS) to store the estimated regression parameters (the
$\beta_j$’s), with the intercept parameter $\beta_0$ at position
B\[$m\,{+}\,1$, 1\] if available

**S**: (default: `" "`) Location (on HDFS) to store the selected feature-ids in the
order as computed by the algorithm, by default it is standard output.

**O**: (default: `" "`) Location (on HDFS) to write certain summary statistics
described in [**Table 9**](algorithms-regression.html#table9), by default it is standard
output.

**link**: (default: `2`) Link function code to determine the link
function $\eta = g(\mu)$, see [**Table 11**](algorithms-regression.html#table11); currently the
following link functions are supported:

  * 1 = log
  * 2 = logit
  * 3 = probit
  * 4 = cloglog

**yneg**: (default: `0.0`) Response value for Bernoulli “No” label, usually 0.0 or -1.0

**icpt**: (default: `0`) Intercept and shifting/rescaling of the features in $X$:

  * 0 = no intercept (hence no $\beta_0$), no
shifting/rescaling of the features
  * 1 = add intercept, but do not shift/rescale the features
in $X$
  * 2 = add intercept, shift/rescale the features in $X$ to
mean 0, variance 1

**tol**: (default: `0.000001`) Tolerance ($\epsilon$) used in the convergence criterion: we
terminate the outer iterations when the deviance changes by less than
this factor; see below for details.

**disp**: (default: `0.0`) Dispersion parameter, or `0.0` to estimate it from
data

**moi**: (default: `200`) Maximum number of outer (Fisher scoring) iterations

**mii**: (default: `0`) Maximum number of inner (conjugate gradient) iterations, or 0
if no maximum limit provided

**thr**: (default: `0.01`) Threshold to stop the algorithm: if the decrease in the value
of the AIC falls below `thr` no further features are being
checked and the algorithm stops.

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.


### Examples

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f StepGLM.dml
                            -nvargs X=/user/ml/X.mtx
                                    Y=/user/ml/Y.mtx
                                    B=/user/ml/B.mtx
                                    S=/user/ml/selected.csv
                                    O=/user/ml/stats.csv
                                    link=2
                                    yneg=-1.0
                                    icpt=2
                                    tol=0.000001
                                    moi=100
                                    mii=10
                                    thr=0.05
                                    fmt=csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f StepGLM.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         Y=/user/ml/Y.mtx
                                         B=/user/ml/B.mtx
                                         S=/user/ml/selected.csv
                                         O=/user/ml/stats.csv
                                         link=2
                                         yneg=-1.0
                                         icpt=2
                                         tol=0.000001
                                         moi=100
                                         mii=10
                                         thr=0.05
                                         fmt=csv
</div>
</div>


### Details

Similar to `StepLinearRegDS.dml` our stepwise GLM script
builds a model by iteratively selecting predictive variables using a
forward selection strategy based on the AIC (5). Note that
currently only the Bernoulli distribution family (`fam=2` in
[**Table 11**](algorithms-regression.html#table11)) together with the following link functions
are supported: `log`, `logit`, `probit`, and `cloglog` (link
$$\in\{1,2,3,4\}$$ in [**Table 11**](algorithms-regression.html#table11)).


### Returns

Similar to the outputs from `GLM.dml` the stepwise GLM script
computes the estimated regression coefficients and stores them in matrix
$B$ on HDFS; matrix $B$ follows the same format as the one produced by
`GLM.dml` (see [Generalized Linear Models](algorithms-regression.html#generalized-linear-models)). Additionally,
`StepGLM.dml` outputs the variable indices (stored in the
1-column matrix $S$) in the order they have been selected by the
algorithm, i.e., $i$th entry in matrix $S$ stores the variable which
improves the AIC the most in $i$th iteration. If the model with the
lowest AIC includes no variables matrix $S$ will be empty. Moreover, the
estimated summary statistics as defined in [**Table 9**](algorithms-regression.html#table9) are
printed out or stored in a file on HDFS (if requested); these statistics
will be provided only if the selected model is nonempty, i.e., contains
at least one variable.


* * *

## 4.5. Regression Scoring and Prediction

### Description

Script `GLM-predict.dml` is intended to cover all linear
model based regressions, including linear regression, binomial and
multinomial logistic regression, and GLM regressions (Poisson, gamma,
binomial with probit link etc.). Having just one scoring script for all
these regressions simplifies maintenance and enhancement while ensuring
compatible interpretations for output statistics.

The script performs two functions, prediction and scoring. To perform
prediction, the script takes two matrix inputs: a collection of records
$X$ (without the response attribute) and the estimated regression
parameters $B$, also known as $\beta$. To perform scoring, in addition
to $X$ and $B$, the script takes the matrix of actual response
values $Y$ that are compared to the predictions made with $X$ and $B$.
Of course there are other, non-matrix, input arguments that specify the
model and the output format, see below for the full list.

We assume that our test/scoring dataset is given by
$n\,{\times}\,m$-matrix $X$ of numerical feature vectors, where each
row $x_i$ represents one feature vector of one record; we have $\dim x_i = m$. Each
record also includes the response variable $y_i$ that may be numerical,
single-label categorical, or multi-label categorical. A single-label
categorical $y_i$ is an integer category label, one label per record; a
multi-label $y_i$ is a vector of integer counts, one count for each
possible label, which represents multiple single-label events
(observations) for the same $x_i$. Internally we convert single-label
categoricals into multi-label categoricals by replacing each label $l$
with an indicator vector $(0,\ldots,0,1_l,0,\ldots,0)$. In
prediction-only tasks the actual $y_i$’s are not needed to the script,
but they are needed for scoring.

To perform prediction, the script matrix-multiplies $X$ and $B$, adding
the intercept if available, then applies the inverse of the model’s link
function. All GLMs assume that the linear combination of the features
in $x_i$ and the betas in $B$ determines the means $\mu_i$ of
the $y_i$’s (in numerical or multi-label categorical form) with
$\dim \mu_i = \dim y_i$. The observed $y_i$ is assumed to follow a
specified GLM family distribution
$Prob[y\mid \mu_i]$
with mean(s) $\mu_i$:

$$x_i \,\,\,\,\mapsto\,\,\,\, \eta_i = \beta_0 + \sum\nolimits_{j=1}^m \beta_j x_{i,j} 
\,\,\,\,\mapsto\,\,\,\, \mu_i \,\,\,\,\mapsto \,\,\,\, y_i \sim Prob[y\mid \mu_i]$$

If $y_i$ is numerical, the predicted mean $\mu_i$ is a real number. Then
our script’s output matrix $M$ is the $n\,{\times}\,1$-vector of these
means $\mu_i$. Note that $\mu_i$ predicts the mean of $y_i$, not the
actual $y_i$. For example, in Poisson distribution, the mean is usually
fractional, but the actual $y_i$ is always integer.

If $y_i$ is categorical, i.e. a vector of label counts for record $i$,
then $\mu_i$ is a vector of non-negative real numbers, one number
$$\mu_{i,l}$$ per each label $l$. In this case we divide the $$\mu_{i,l}$$
by their sum $\sum_l \mu_{i,l}$ to obtain predicted label
probabilities $$p_{i,l}\in [0, 1]$$. The output matrix $M$ is the
$n \times (k\,{+}\,1)$-matrix of these probabilities, where $n$ is the
number of records and $k\,{+}\,1$ is the number of categories[^3]. Note
again that we do not predict the labels themselves, nor their actual
counts per record, but we predict the labels’ probabilities.

Going from predicted probabilities to predicted labels, in the
single-label categorical case, requires extra information such as the
cost of false positive versus false negative errors. For example, if
there are 5 categories and we *accurately* predicted their probabilities
as $(0.1, 0.3, 0.15, 0.2, 0.25)$, just picking the highest-probability
label would be wrong 70% of the time, whereas picking the
lowest-probability label might be right if, say, it represents a
diagnosis of cancer or another rare and serious outcome. Hence, we keep
this step outside the scope of `GLM-predict.dml` for now.


### Usage

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs X=<file>
                                    Y=[file]
                                    B=<file>
                                    M=[file]
                                    O=[file]
                                    dfam=[int]
                                    vpow=[double]
                                    link=[int]
                                    lpow=[double]
                                    disp=[double]
                                    fmt=[format]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         Y=[file]
                                         B=<file>
                                         M=[file]
                                         O=[file]
                                         dfam=[int]
                                         vpow=[double]
                                         link=[int]
                                         lpow=[double]
                                         disp=[double]
                                         fmt=[format]
</div>
</div>


### Arguments

**X**: Location (on HDFS) to read the $n\,{\times}\,m$-matrix $X$ of feature
vectors, each row constitutes one feature vector (one record)

**Y**: (default: `" "`) Location to read the response matrix $Y$ needed for scoring
(but optional for prediction), with the following dimensions:

  * $n {\times} 1$: acceptable for all distributions
(`dfam=1` or `2` or `3`)
  * $n {\times} 2$: for binomial (`dfam=2`) if given by
(\#pos, \#neg) counts
  * $n {\times} k\,{+}\,1$: for multinomial (`dfam=3`) if
given by category counts

**M**: (default: `" "`) Location to write, if requested, the matrix of predicted
response means (for `dfam=1`) or probabilities (for
`dfam=2` or `3`):

  * $n {\times} 1$: for power-type distributions (`dfam=1`)
  * $n {\times} 2$: for binomial distribution (`dfam=2`),
col\# 2 is the “No” probability
  * $n {\times} k\,{+}\,1$: for multinomial logit (`dfam=3`),
col\# $k\,{+}\,1$ is for the baseline

**B**: Location to read matrix $B$ of the betas, i.e. estimated GLM regression
parameters, with the intercept at row\# $m\,{+}\,1$ if available:

  * $\dim(B) \,=\, m {\times} k$: do not add intercept
  * $\dim(B) \,=\, m\,{+}\,1 {\times} k$: add intercept as given by the
last $B$-row
  * if $k > 1$, use only $B[, 1]$ unless it is Multinomial Logit
(`dfam=3`)

**O**: (default: `" "`) Location to store the CSV-file with goodness-of-fit
statistics defined in [**Table 13**](algorithms-regression.html#table13), 
the default is to
print them to the standard output

**dfam**: (default: `1`) GLM distribution family code to specify the type of
distribution
$Prob[y\,|\,\mu]$
that we assume:

  * 1 = power distributions with
$Var(y) = \mu^{\alpha}$, see
[**Table 11**](algorithms-regression.html#table11)
  * 2 = binomial
  * 3 = multinomial logit

**vpow**: (default: `0.0`) Power for variance defined as (mean)$^{\textrm{power}}$
(ignored if `dfam`$\,{\neq}\,1$): when `dfam=1`,
this provides the $q$ in
$Var(y) = a\mu^q$, the power
dependence of the variance of $y$ on its mean. In particular, use:

  * 0.0 = Gaussian
  * 1.0 = Poisson
  * 2.0 = Gamma
  * 3.0 = inverse Gaussian

**link**: (default: `0`) Link function code to determine the link
function $\eta = g(\mu)$, ignored for multinomial logit
(`dfam=3`):

  * 0 = canonical link (depends on the distribution family),
see [**Table 11**](algorithms-regression.html#table11)
  * 1 = power functions
  * 2 = logit
  * 3 = probit
  * 4 = cloglog
  * 5 = cauchit

**lpow**: (default: `1.0`) Power for link function defined as
(mean)$^{\textrm{power}}$ (ignored if `link`$\,{\neq}\,1$):
when `link=1`, this provides the $s$ in $\eta = \mu^s$, the
power link function; `lpow=0.0` gives the log link
$\eta = \log\mu$. Common power links:

  * -2.0 = $1/\mu^2$ 
  * -1.0 = reciprocal
  * 0.0 = log 
  * 0.5 = sqrt 
  * 1.0 = identity

**disp**: (default: `1.0`) Dispersion value, when available; must be positive

**fmt**: (default: `"text"`) Matrix M file output format, such as
`text`, `mm`, or `csv`; see read/write
functions in SystemML Language Reference for details.


### Examples

Note that in the examples below the value for the `disp` input
argument is set arbitrarily. The correct dispersion value should be
computed from the training data during model estimation, or omitted if
unknown (which sets it to `1.0`).

**Linear regression example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=1
                                    vpow=0.0
                                    link=1
                                    lpow=1.0
                                    disp=5.67
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Means.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=1
                                         vpow=0.0
                                         link=1
                                         lpow=1.0
                                         disp=5.67
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Means.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

**Linear regression example, prediction only (no Y given)**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=1
                                    vpow=0.0
                                    link=1
                                    lpow=1.0
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Means.mtx
                                    fmt=csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=1
                                         vpow=0.0
                                         link=1
                                         lpow=1.0
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Means.mtx
                                         fmt=csv
</div>
</div>

**Binomial logistic regression example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=2
                                    link=2
                                    disp=3.0004464
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Probabilities.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=2
                                         link=2
                                         disp=3.0004464
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Probabilities.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

**Binomial probit regression example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=2
                                    link=3
                                    disp=3.0004464
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Probabilities.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=2
                                         link=3
                                         disp=3.0004464
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Probabilities.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

**Multinomial logistic regression example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=3 
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Probabilities.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=3
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Probabilities.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

**Poisson regression with the log link example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=1
                                    vpow=1.0
                                    link=1
                                    lpow=0.0
                                    disp=3.45
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Means.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=1
                                         vpow=1.0
                                         link=1
                                         lpow=0.0
                                         disp=3.45
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Means.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

**Gamma regression with the inverse (reciprocal) link example**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f GLM-predict.dml
                            -nvargs dfam=1
                                    vpow=2.0
                                    link=1
                                    lpow=-1.0
                                    disp=1.99118
                                    X=/user/ml/X.mtx
                                    B=/user/ml/B.mtx
                                    M=/user/ml/Means.mtx
                                    fmt=csv
                                    Y=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f GLM-predict.dml
                                 -config=SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs dfam=1
                                         vpow=2.0
                                         link=1
                                         lpow=-1.0
                                         disp=1.99118
                                         X=/user/ml/X.mtx
                                         B=/user/ml/B.mtx
                                         M=/user/ml/Means.mtx
                                         fmt=csv
                                         Y=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

* * *

<a name="table13" />
**Table 13**: The goodness-of-fit statistics are provided in CSV format, one per each line, with four
columns: (Name, CID, Disp?, Value).  The columns are: 
“Name” is the string identifier for the statistic;
“CID” is an optional integer value that specifies the $Y$-column index for per-column statistics
(note that a bi/multinomial one-column Y-input is converted into multi-column);
“Disp?” is an optional Boolean value ($TRUE$ or $FALSE$) that tells us
whether or not scaling by the input dispersion parameter `disp` has been applied to this
statistic;
“Value” is the value of the statistic.

| Name                 | CID | Disp? | Meaning |
| -------------------- | :-: | :---: | ------- |
| LOGLHOOD\_Z          |     |   +   | Log-likelihood $Z$-score (in st. dev.'s from the mean) |
| LOGLHOOD\_Z\_PVAL    |     |   +   | Log-likelihood $Z$-score p-value, two-sided |
| PEARSON\_X2          |     |   +   | Pearson residual $X^2$-statistic |
| PEARSON\_X2\_BY\_DF  |     |   +   | Pearson $X^2$ divided by degrees of freedom |
| PEARSON\_X2\_PVAL    |     |   +   | Pearson $X^2$ p-value |
| DEVIANCE\_G2         |     |   +   | Deviance from the saturated model $G^2$-statistic |
| DEVIANCE\_G2\_BY\_DF |     |   +   | Deviance $G^2$ divided by degrees of freedom |
| DEVIANCE\_G2\_PVAL   |     |   +   | Deviance $G^2$ p-value |
| AVG\_TOT\_Y          |  +  |       | $Y$-column average for an individual response value |
| STDEV\_TOT\_Y        |  +  |       | $Y$-column st. dev. for an individual response value |
| AVG\_RES\_Y          |  +  |       | $Y$-column residual average of $Y - pred. mean(Y\\|X)$ |
| STDEV\_RES\_Y        |  +  |       | $Y$-column residual st. dev. of $Y - pred. mean(Y\\|X)$ |
| PRED\_STDEV\_RES     |  +  |   +   | Model-predicted $Y$-column residual st. deviation|
| PLAIN\_R2            |  +  |       | Plain $R^2$ of $Y$-column residual with bias included |
| ADJUSTED\_R2         |  +  |       | Adjusted $R^2$ of $Y$-column residual w. bias included |
| PLAIN\_R2\_NOBIAS    |  +  |       | Plain $R^2$ of $Y$-column residual, bias subtracted |
| ADJUSTED\_R2\_NOBIAS |  +  |       | Adjusted $R^2$ of $Y$-column residual, bias subtracted |

* * *


### Details

The output matrix $M$ of predicted means (or probabilities) is computed
by matrix-multiplying $X$ with the first column of $B$ or with the
whole $B$ in the multinomial case, adding the intercept if available
(conceptually, appending an extra column of ones to $X$); then applying
the inverse of the model’s link function. The difference between “means”
and “probabilities” in the categorical case becomes significant when
there are ${\geq}\,2$ observations per record (with the multi-label
records) or when the labels such as $-1$ and $1$ are viewed and averaged
as numerical response values (with the single-label records). To avoid
any or information loss, we separately return the predicted probability
of each category label for each record.

When the “actual” response values $Y$ are available, the summary
statistics are computed and written out as described in
[**Table 13**](algorithms-regression.html#table13). Below we discuss each of these statistics
in detail. Note that in the categorical case (binomial and multinomial)
$Y$ is internally represented as the matrix of observation counts for
each label in each record, rather than just the label ID for each
record. The input $Y$ may already be a matrix of counts, in which case
it is used as-is. But if $Y$ is given as a vector of response labels,
each response label is converted into an indicator vector
$(0,\ldots,0,1_l,0,\ldots,0)$ where $l$ is the label ID for this record.
All negative (e.g. $-1$) or zero label IDs are converted to the
$1 +$ maximum label ID. The largest label ID is viewed as the
“baseline” as explained in the section on Multinomial Logistic
Regression. We assume that there are $k\geq 1$ non-baseline categories
and one (last) baseline category.

We also estimate residual variances for each response value, although we
do not output them, but use them only inside the summary statistics,
scaled and unscaled by the input dispersion parameter `disp`,
as described below.

`LOGLHOOD_Z` and `LOGLHOOD_Z_PVAL` statistics measure how far the
log-likelihood of $Y$ deviates from its expected value according to the
model. The script implements them only for the binomial and the
multinomial distributions, returning `NaN` for all other distributions.
Pearson’s $X^2$ and deviance $G^2$ often perform poorly with bi- and
multinomial distributions due to low cell counts, hence we need this
extra goodness-of-fit measure. To compute these statistics, we use:

  * the $n\times (k\,{+}\,1)$-matrix $Y$ of multi-label response counts, in
which $y_{i,j}$ is the number of times label $j$ was observed in
record $i$
  * the model-estimated probability matrix $P$ of the same dimensions that
satisfies $$\sum_{j=1}^{k+1} p_{i,j} = 1$$ for all $i=1,\ldots,n$ and
where $p_{i,j}$ is the model probability of observing label $j$ in
record $i$
  * the $n\,{\times}\,1$-vector $N$ where $N_i$ is the aggregated count of
observations in record $i$ (all $N_i = 1$ if each record has only one
response label)

We start by computing the multinomial log-likelihood of $Y$ given $P$
and $N$, as well as the expected log-likelihood given a random $Y$ and
the variance of this log-likelihood if $Y$ indeed follows the proposed
distribution: 

$$
\begin{aligned}
\ell (Y) \,\,&=\,\, \log Prob[Y \,|\, P, N] \,\,=\,\, \sum_{i=1}^{n} \,\sum_{j=1}^{k+1}  \,y_{i,j}\log p_{i,j} \\
E_Y \ell (Y)  \,\,&=\,\, \sum_{i=1}^{n}\, \sum_{j=1}^{k+1} \,\mu_{i,j} \log p_{i,j} 
    \,\,=\,\, \sum_{i=1}^{n}\, N_i \,\sum_{j=1}^{k+1} \,p_{i,j} \log p_{i,j} \\
Var_Y \ell (Y) \,&=\, \sum_{i=1}^{n} \,N_i \left(\sum_{j=1}^{k+1} \,p_{i,j} \big(\log p_{i,j}\big)^2
    - \Bigg( \sum_{j=1}^{k+1} \,p_{i,j} \log p_{i,j}\Bigg) ^ {\!\!2\,} \right)
\end{aligned}
$$


Then we compute the $Z$-score as the difference between the actual and
the expected log-likelihood $\ell(Y)$ divided by its expected standard
deviation, and its two-sided p-value in the Normal distribution
assumption ($\ell(Y)$ should approach normality due to the Central Limit
Theorem):

$$
Z \,=\, \frac {\ell(Y) - E_Y \ell(Y)}{\sqrt{Var_Y \ell(Y)}};\quad
\mathop{\textrm{p-value}}(Z) \,=\, Prob\Big[\,\big|\mathop{\textrm{Normal}}(0,1)\big| \, > \, |Z|\,\Big]
$$

A low p-value would indicate “underfitting” if $Z\ll 0$ or “overfitting”
if $Z\gg 0$. Here “overfitting” means that higher-probability labels
occur more often than their probabilities suggest.

We also apply the dispersion input (`disp`) to compute the
“scaled” version of the $Z$-score and its p-value. Since $\ell(Y)$ is a
linear function of $Y$, multiplying the GLM-predicted variance of $Y$ by
`disp` results in multiplying
$Var_Y \ell(Y)$ by the same
`disp`. This, in turn, translates into dividing the $Z$-score
by the square root of the dispersion:

$$Z_{\texttt{disp}}  \,=\, \big(\ell(Y) \,-\, E_Y \ell(Y)\big) \,\big/\, \sqrt{\texttt{disp}\cdot Var_Y \ell(Y)}
\,=\, Z / \sqrt{\texttt{disp}}$$ 

Finally, we recalculate the p-value
with this new $Z$-score.

`PEARSON_X2`, `PEARSON_X2_BY_DF`, and `PEARSON_X2_PVAL`:
Pearson’s residual $X^2$-statistic is a commonly used goodness-of-fit
measure for linear models [[McCullagh1989]](algorithms-bibliography.html).
The idea is to measure how
well the model-predicted means and variances match the actual behavior
of response values. For each record $i$, we estimate the mean $\mu_i$
and the variance $v_i$ (or `disp` $\cdot v_i$) and use them to
normalize the residual: $r_i = (y_i - \mu_i) / \sqrt{v_i}$. These
normalized residuals are then squared, aggregated by summation, and
tested against an appropriate $\chi^2$ distribution. The computation
of $X^2$ is slightly different for categorical data (bi- and
multinomial) than it is for numerical data, since $y_i$ has multiple
correlated dimensions [[McCullagh1989]](algorithms-bibliography.html):

$$X^2\,\textrm{(numer.)} \,=\,  \sum_{i=1}^{n}\, \frac{(y_i - \mu_i)^2}{v_i};\quad
X^2\,\textrm{(categ.)} \,=\,  \sum_{i=1}^{n}\, \sum_{j=1}^{k+1} \,\frac{(y_{i,j} - N_i 
\hspace{0.5pt} p_{i,j})^2}{N_i \hspace{0.5pt} p_{i,j}}$$ 

The number of
degrees of freedom \#d.f. for the $\chi^2$ distribution is $n - m$ for
numerical data and $(n - m)k$ for categorical data, where
$k = \mathop{\texttt{ncol}}(Y) - 1$. Given the dispersion parameter
`disp` the $X^2$ statistic is scaled by division: $$X^2_{\texttt{disp}} = X^2 / \texttt{disp}$$. If the
dispersion is accurate, $X^2 / \texttt{disp}$ should be close to \#d.f.
In fact, $X^2 / \textrm{\#d.f.}$ over the *training* data is the
dispersion estimator used in our `GLM.dml` script,
see (7). Here we provide $X^2 / \textrm{\#d.f.}$ and
$X^2_{\texttt{disp}} / \textrm{\#d.f.}$ as
`PEARSON_X2_BY_DF` to enable dispersion comparison between
the training data and the test data.

NOTE: For categorical data, both Pearson’s $X^2$ and the deviance $G^2$
are unreliable (i.e. do not approach the $\chi^2$ distribution) unless
the predicted means of multi-label counts
$$\mu_{i,j} = N_i \hspace{0.5pt} p_{i,j}$$ are fairly large: all
${\geq}\,1$ and 80% are at least $5$ [[Cochran1954]](algorithms-bibliography.html). They should not
be used for “one label per record” categoricals.

`DEVIANCE_G2`, `DEVIANCE_G2_BY_DF`, and
`DEVIANCE_G2_PVAL`: Deviance $G^2$ is the log of the
likelihood ratio between the “saturated” model and the linear model
being tested for the given dataset, multiplied by two:

$$
\begin{equation}
G^2 \,=\, 2 \,\log \frac{Prob[Y \mid \textrm{saturated model}\hspace{0.5pt}]}{Prob[Y \mid \textrm{tested linear model}\hspace{0.5pt}]}
\end{equation}
$$

The “saturated” model sets the mean
$\mu_i^{\mathrm{sat}}$ to equal $y_i$ for every record (for categorical
data, $$p_{i,j}^{sat} = y_{i,j} / N_i$$), which represents the
“perfect fit.” For records with $y_{i,j} \in \{0, N_i\}$ or otherwise at
a boundary, by continuity we set $0 \log 0 = 0$. The GLM likelihood
functions defined in (6) become simplified in
ratio (8) due to canceling out the term $c(y, a)$
since it is the same in both models.

The log of a likelihood ratio between two nested models, times two, is
known to approach a $\chi^2$ distribution as $n\to\infty$ if both models
have fixed parameter spaces. But this is not the case for the
“saturated” model: it adds more parameters with each record. In
practice, however, $\chi^2$ distributions are used to compute the
p-value of $G^2$ [[McCullagh1989]](algorithms-bibliography.html). The number of degrees of
freedom \#d.f. and the treatment of dispersion are the same as for
Pearson’s $X^2$, see above.


### Column-Wise Statistics

The rest of the statistics are computed separately for each column
of $Y$. As explained above, $Y$ has two or more columns in bi- and
multinomial case, either at input or after conversion. Moreover, each
$$y_{i,j}$$ in record $i$ with $N_i \geq 2$ is counted as $N_i$ separate
observations $$y_{i,j,l}$$ of 0 or 1 (where $l=1,\ldots,N_i$) with
$$y_{i,j}$$ ones and $$N_i-y_{i,j}$$ zeros. For power distributions,
including linear regression, $Y$ has only one column and all $N_i = 1$,
so the statistics are computed for all $Y$ with each record counted
once. Below we denote $$N = \sum_{i=1}^n N_i \,\geq n$$. Here is the total
average and the residual average (residual bias) of $y_{i,j,l}$ for each
$Y$-column:

$$\texttt{AVG_TOT_Y}_j   \,=\, \frac{1}{N} \sum_{i=1}^n  y_{i,j}; \quad
\texttt{AVG_RES_Y}_j   \,=\, \frac{1}{N} \sum_{i=1}^n \, (y_{i,j} - \mu_{i,j})$$

Dividing by $N$ (rather than $n$) gives the averages for $$y_{i,j,l}$$
(rather than $$y_{i,j}$$). The total variance, and the standard deviation,
for individual observations $$y_{i,j,l}$$ is estimated from the total
variance for response values $$y_{i,j}$$ using independence assumption:
$$Var \,y_{i,j} = Var \sum_{l=1}^{N_i} y_{i,j,l} = \sum_{l=1}^{N_i} Var y_{i,j,l}$$.
This allows us to estimate the sum of squares for $y_{i,j,l}$ via the
sum of squares for $$y_{i,j}$$: 

$$\texttt{STDEV_TOT_Y}_j \,=\, 
\Bigg[\frac{1}{N-1} \sum_{i=1}^n  \Big( y_{i,j} -  \frac{N_i}{N} \sum_{i'=1}^n  y_{i'\!,j}\Big)^2\Bigg]^{1/2}$$

Analogously, we estimate the standard deviation of the residual
$$y_{i,j,l} - \mu_{i,j,l}$$: 

$$\texttt{STDEV_RES_Y}_j \,=\, 
\Bigg[\frac{1}{N-m'} \,\sum_{i=1}^n  \Big( y_{i,j} - \mu_{i,j} -  \frac{N_i}{N} \sum_{i'=1}^n  (y_{i'\!,j} - \mu_{i'\!,j})\Big)^2\Bigg]^{1/2}$$

Here $m'=m$ if $m$ includes the intercept as a feature and $m'=m+1$ if
it does not. The estimated standard deviations can be compared to the
model-predicted residual standard deviation computed from the predicted
means by the GLM variance formula and scaled by the dispersion:

$$\texttt{PRED_STDEV_RES}_j \,=\, \Big[\frac{\texttt{disp}}{N} \, \sum_{i=1}^n \, v(\mu_{i,j})\Big]^{1/2}$$

We also compute the $R^2$ statistics for each column of $Y$, see
[**Table 14**](algorithms-regression.html#table14) and [**Table 15**](algorithms-regression.html#table15) for details. We compute two versions
of $R^2$: in one version the residual sum-of-squares (RSS) includes any
bias in the residual that might be present (due to the lack of, or
inaccuracy in, the intercept); in the other version of RSS the bias is
subtracted by “centering” the residual. In both cases we subtract the
bias in the total sum-of-squares (in the denominator), and $m'$ equals
$m$ with the intercept or $m+1$ without the intercept.


* * *

<a name="table14" />
**Table 14**: $R^2$ where the residual sum-of-squares includes the bias contribution.

| Statistic             | Formula |
| --------------------- | ------------- |
| $\texttt{PLAIN_R2}_j$ | $$ \displaystyle 1 - \frac{\sum\limits_{i=1}^n \,(y_{i,j} - \mu_{i,j})^2}{\sum\limits_{i=1}^n \Big(y_{i,j} - \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n  y_{i',j} \Big)^{2}} $$
| $\texttt{ADJUSTED_R2}_j$ | $$ \displaystyle 1 - {\textstyle\frac{N_{\mathstrut} - 1}{N^{\mathstrut} - m}}  \, \frac{\sum\limits_{i=1}^n \,(y_{i,j} - \mu_{i,j})^2}{\sum\limits_{i=1}^n \Big(y_{i,j} - \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n  y_{i',j} \Big)^{2}} $$
 

* * *

<a name="table15" />
**Table 15**: $R^2$ where the residual sum-of-squares is centered so that the bias is subtracted.

| Statistic             | Formula |
| --------------------- | ------------- |
| $\texttt{PLAIN_R2_NOBIAS}_j$ | $$ \displaystyle 1 - \frac{\sum\limits_{i=1}^n \Big(y_{i,j} \,{-}\, \mu_{i,j} \,{-}\, \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n  (y_{i',j} \,{-}\, \mu_{i',j}) \Big)^{2}}{\sum\limits_{i=1}^n \Big(y_{i,j} - \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n y_{i',j} \Big)^{2}} $$
| $\texttt{ADJUSTED_R2_NOBIAS}_j$ | $$ \displaystyle 1 - {\textstyle\frac{N_{\mathstrut} - 1}{N^{\mathstrut} - m'}} \, \frac{\sum\limits_{i=1}^n \Big(y_{i,j} \,{-}\, \mu_{i,j} \,{-}\, \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n  (y_{i',j} \,{-}\, \mu_{i',j}) \Big)^{2}}{\sum\limits_{i=1}^n \Big(y_{i,j} - \frac{N_{i\mathstrut}}{N^{\mathstrut}} \sum\limits_{i'=1}^n y_{i',j} \Big)^{2}} $$


* * *


### Returns

The matrix of predicted means (if the response is numerical) or
probabilities (if the response is categorical), see Description
subsection above for more information. Given `Y`, we return
some statistics in CSV format as described in
[**Table 13**](algorithms-regression.html#table13) and in the above text.


* * *


[^1]: Smaller likelihood difference between two models suggests less
    statistical evidence to pick one model over the other.

[^2]: $Prob[y\mid \mu_i]$
    is given by a density function if $y$ is continuous.

[^3]: We use $k+1$ because there are $k$ non-baseline categories and one
    baseline category, with regression parameters $B$ having
    $k$ columns.
