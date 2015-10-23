---
layout: global
title: SystemML Algorithms Reference - Survival Analysis
displayTitle: <a href="algorithms-reference.html">SystemML Algorithms Reference</a>
---


# 6. Survival Analysis

## 6.1. Kaplan-Meier Survival Analysis

### Description

Survival analysis examines the time needed for a particular event of
interest to occur. In medical research, for example, the prototypical
such event is the death of a patient but the methodology can be applied
to other application areas, e.g., completing a task by an individual in
a psychological experiment or the failure of electrical components in
engineering. Kaplan-Meier or (product limit) method is a simple
non-parametric approach for estimating survival probabilities from both
censored and uncensored survival times.


### Usage

    hadoop jar SystemML.jar -f KM.dml
                            -nvargs X=<file>
                                    TE=<file>
                                    GI=<file>
                                    SI=<file>
                                    O=<file>
                                    M=<file>
                                    T=<file>
                                    alpha=[double]
                                    etype=[greenwood|peto]
                                    ctype=[plain|log|log-log]
                                    ttype=[none|log-rank|wilcoxon]
                                    fmt=[format]


### Arguments

**X**: Location (on HDFS) to read the input matrix of the survival data
containing:

  * timestamps
  * whether event occurred (1) or data is censored (0)
  * a number of factors (i.e., categorical features) for grouping and/or
stratifying

**TE**: Location (on HDFS) to read the 1-column matrix $TE$ that contains the
column indices of the input matrix $X$ corresponding to timestamps
(first entry) and event information (second entry)

**GI**: Location (on HDFS) to read the 1-column matrix $GI$ that contains the
column indices of the input matrix $X$ corresponding to the factors
(i.e., categorical features) to be used for grouping

**SI**: Location (on HDFS) to read the 1-column matrix $SI$ that contains the
column indices of the input matrix $X$ corresponding to the factors
(i.e., categorical features) to be used for grouping

**O**: Location (on HDFS) to write the matrix containing the results of the
Kaplan-Meier analysis $KM$

**M**: Location (on HDFS) to write Matrix $M$ containing the following
statistics: total number of events, median and its confidence intervals;
if survival data for multiple groups and strata are provided each row of
$M$ contains the above statistics per group and stratum.

**T**: If survival data from multiple groups is available and
`ttype=log-rank` or `ttype=wilcoxon`, location (on
HDFS) to write the two matrices that contains the result of the
(stratified) test for comparing these groups; see below for details.

**alpha**: (default: `0.05`) Parameter to compute $100(1-\alpha)\%$ confidence intervals
for the survivor function and its median

**etype**: (default: `"greenwood"`) Parameter to specify the error type according to `greenwood`
or `peto`

**ctype**: (default: `"log"`) Parameter to modify the confidence interval; `plain` keeps
the lower and upper bound of the confidence interval unmodified, `log`
corresponds to logistic transformation and `log-log` corresponds to the
complementary log-log transformation

**ttype**: (default: `"none"`) If survival data for multiple groups is available specifies
which test to perform for comparing survival data across multiple
groups: `none`, `log-rank` or `wilcoxon` test

**fmt**: (default:`"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.


### Examples

    hadoop jar SystemML.jar -f KM.dml
                            -nvargs X=/user/ml/X.mtx
                                    TE=/user/ml/TE
                                    GI=/user/ml/GI
                                    SI=/user/ml/SI
                                    O=/user/ml/kaplan-meier.csv
                                    M=/user/ml/model.csv
                                    alpha=0.01
                                    etype=greenwood
                                    ctype=plain
                                    fmt=csv

    hadoop jar SystemML.jar -f KM.dml
                            -nvargs X=/user/ml/X.mtx
                                    TE=/user/ml/TE
                                    GI=/user/ml/GI
                                    SI=/user/ml/SI
                                    O=/user/ml/kaplan-meier.csv
                                    M=/user/ml/model.csv
                                    T=/user/ml/test.csv
                                    alpha=0.01
                                    etype=peto
                                    ctype=log
                                    ttype=log-rank
                                    fmt=csv


### Details

The Kaplan-Meier estimate is a non-parametric maximum likelihood
estimate (MLE) of the survival function $S(t)$, i.e., the probability of
survival from the time origin to a given future time. As an illustration
suppose that there are $n$ individuals with observed survival times
$t_1,t_2,\ldots t_n$ out of which there are $r\leq n$ distinct death
times $$t_{(1)}\leq t_{(2)}\leq t_{(r)}$$—since some of the observations
may be censored, in the sense that the end-point of interest has not
been observed for those individuals, and there may be more than one
individual with the same survival time. Let $S(t_j)$ denote the
probability of survival until time $t_j$, $d_j$ be the number of events
at time $t_j$, and $n_j$ denote the number of individual at risk (i.e.,
those who die at time $t_j$ or later). Assuming that the events occur
independently, in Kaplan-Meier method the probability of surviving from
$t_j$ to $$t_{j+1}$$ is estimated from $S(t_j)$ and given by

$$\hat{S}(t) = \prod_{j=1}^{k} \left( \frac{n_j-d_j}{n_j} \right)$$

for
$$t_k\leq t<t_{k+1}$$, $$k=1,2,\ldots r$$, $$\hat{S}(t)=1$$ for $$t<t_{(1)}$$,
and $$t_{(r+1)}=\infty$$. Note that the value of $\hat{S}(t)$ is constant
between times of event and therefore the estimate is a step function
with jumps at observed event times. If there are no censored data this
estimator would simply reduce to the empirical survivor function defined
as $\frac{n_j}{n}$. Thus, the Kaplan-Meier estimate can be seen as the
generalization of the empirical survivor function that handles censored
observations.

The methodology used in our `KM.dml` script closely
follows Section 2 of [[Collett2003]](algorithms-bibliography.html). For completeness we briefly
discuss the equations used in our implementation.

**Standard error of the survivor function.** The standard error of the
estimated survivor function (controlled by parameter `etype`)
can be calculated as

$$\text{se} \{\hat{S}(t)\} \approx \hat{S}(t) {\bigg\{ \sum_{j=1}^{k} \frac{d_j}{n_j(n_j -   d_j)}\biggr\}}^2$$

for $$t_{(k)}\leq t<t_{(k+1)}$$. This equation is known as the
*Greenwood’s* formula. An alternative approach is to apply
the *Petos’s* expression

$$\text{se}\{\hat{S}(t)\}=\frac{\hat{S}(t)\sqrt{1-\hat{S}(t)}}{\sqrt{n_k}}$$

for $$t_{(k)}\leq t<t_{(k+1)}$$. Once the standard error of $\hat{S}$ has
been found we compute the following types of confidence intervals
(controlled by parameter `cctype`): The `plain`
$100(1-\alpha)\%$ confidence interval for $S(t)$ is computed using

$$\hat{S}(t)\pm z_{\alpha/2} \text{se}\{\hat{S}(t)\}$$

where
$z_{\alpha/2}$ is the upper $\alpha/2$-point of the standard normal
distribution. Alternatively, we can apply the `log` transformation using

$$\hat{S}(t)^{\exp[\pm z_{\alpha/2} \text{se}\{\hat{S}(t)\}/\hat{S}(t)]}$$

or the `log-log` transformation using

$$\hat{S}(t)^{\exp [\pm z_{\alpha/2} \text{se} \{\log [-\log \hat{S}(t)]\}]}$$

**Median, its standard error and confidence interval.** Denote by
$$\hat{t}(50)$$ the estimated median of $$\hat{S}$$, i.e.,
$$\hat{t}(50)=\min \{ t_i \mid \hat{S}(t_i) < 0.5\}$$, where $$t_i$$ is the
observed survival time for individual $$i$$. The standard error of
$$\hat{t}(50)$$ is given by

$$\text{se}\{ \hat{t}(50) \} = \frac{1}{\hat{f}\{\hat{t}(50)\}} \text{se}[\hat{S}\{ \hat{t}(50) \}]$$

where $$\hat{f}\{ \hat{t}(50) \}$$ can be found from

$$\hat{f}\{ \hat{t}(50) \} = \frac{\hat{S}\{ \hat{u}(50) \} -\hat{S}\{ \hat{l}(50) \} }{\hat{l}(50) - \hat{u}(50)}$$

Above, $\hat{u}(50)$ is the largest survival time for which $\hat{S}$
exceeds $0.5+\epsilon$, i.e.,
$$\hat{u}(50)=\max \bigl\{ t_{(j)} \mid \hat{S}(t_{(j)}) \geq 0.5+\epsilon \bigr\}$$,
and $\hat{l}(50)$ is the smallest survivor time for which $\hat{S}$ is
less than $0.5-\epsilon$, i.e.,
$$\hat{l}(50)=\min \bigl\{ t_{(j)} \mid \hat{S}(t_{(j)}) \leq 0.5+\epsilon \bigr\}$$,
for small $\epsilon$.

**Log-rank test and Wilcoxon test.** Our implementation supports
comparison of survival data from several groups using two non-parametric
procedures (controlled by parameter `ttype`): the
*log-rank test* and the *Wilcoxon test* (also
known as the *Breslow test*). Assume that the survival
times in $g\geq 2$ groups of survival data are to be compared. Consider
the *null hypothesis* that there is no difference in the
survival times of the individuals in different groups. One way to
examine the null hypothesis is to consider the difference between the
observed number of deaths with the numbers expected under the null
hypothesis. In both tests we define the $U$-statistics ($$U_{L}$$ for the
log-rank test and $$U_{W}$$ for the Wilcoxon test) to compare the observed
and the expected number of deaths in $1,2,\ldots,g-1$ groups as follows:

$$\begin{aligned}
U_{Lk} &= \sum_{j=1}^{r}\left( d_{kj} - \frac{n_{kj}d_j}{n_j} \right) \\
U_{Wk} &= \sum_{j=1}^{r}n_j\left( d_{kj} - \frac{n_{kj}d_j}{n_j} \right)\end{aligned}$$

where $$d_{kj}$$ is the of number deaths at time $$t_{(j)}$$ in group $k$,
$$n_{kj}$$ is the number of individuals at risk at time $$t_{(j)}$$ in group
$k$, and $k=1,2,\ldots,g-1$ to form the vectors $U_L$ and $U_W$ with
$(g-1)$ components. The covariance (variance) between $$U_{Lk}$$ and
$$U_{Lk'}$$ (when $k=k'$) is computed as

$$V_{Lkk'}=\sum_{j=1}^{r} \frac{n_{kj}d_j(n_j-d_j)}{n_j(n_j-1)} \left( \delta_{kk'}-\frac{n_{k'j}}{n_j} \right)$$

for $k,k'=1,2,\ldots,g-1$, with

$$\delta_{kk'} = 
\begin{cases}
1 & \text{if } k=k'\\
0 & \text{otherwise}
\end{cases}$$

These terms are combined in a
*variance-covariance* matrix $V_L$ (referred to as the
$V$-statistic). Similarly, the variance-covariance matrix for the
Wilcoxon test $V_W$ is a matrix where the entry at position $(k,k')$ is
given by

$$V_{Wkk'}=\sum_{j=1}^{r} n_j^2 \frac{n_{kj}d_j(n_j-d_j)}{n_j(n_j-1)} \left( \delta_{kk'}-\frac{n_{k'j}}{n_j} \right)$$

Under the null hypothesis of no group differences, the test statistics
$U_L^\top V_L^{-1} U_L$ for the log-rank test and
$U_W^\top V_W^{-1} U_W$ for the Wilcoxon test have a Chi-squared
distribution on $(g-1)$ degrees of freedom. Our `KM.dml`
script also provides a stratified version of the log-rank or Wilcoxon
test if requested. In this case, the values of the $U$- and $V$-
statistics are computed for each stratum and then combined over all
strata.

### Returns


Below we list the results of the survival analysis computed by
`KM.dml`. The calculated statistics are stored in matrix $KM$
with the following schema:

  * Column 1: timestamps
  * Column 2: number of individuals at risk
  * Column 3: number of events
  * Column 4: Kaplan-Meier estimate of the survivor function $\hat{S}$
  * Column 5: standard error of $\hat{S}$
  * Column 6: lower bound of $100(1-\alpha)\%$ confidence interval for
    $\hat{S}$
  * Column 7: upper bound of $100(1-\alpha)\%$ confidence interval for
    $\hat{S}$

Note that if survival data for multiple groups and/or strata is
available, each collection of 7 columns in $KM$ stores the results per
group and/or per stratum. In this case $KM$ has $7g+7s$ columns, where
$g\geq 1$ and $s\geq 1$ denote the number of groups and strata,
respectively.

Additionally, `KM.dml` stores the following statistics in the
1-row matrix $M$ whose number of columns depends on the number of groups
($g$) and strata ($s$) in the data. Below $k$ denotes the number of
factors used for grouping and $l$ denotes the number of factors used for
stratifying.

  * Columns 1 to $k$: unique combination of values in the $k$ factors
    used for grouping
  * Columns $k+1$ to $k+l$: unique combination of values in the $l$
    factors used for stratifying
  * Column $k+l+1$: total number of records
  * Column $k+l+2$: total number of events
  * Column $k+l+3$: median of $\hat{S}$
  * Column $k+l+4$: lower bound of $100(1-\alpha)\%$ confidence interval
    for the median of $\hat{S}$
  * Column $k+l+5$: upper bound of $100(1-\alpha)\%$ confidence interval
    for the median of $\hat{S}$.

If there is only 1 group and 1 stratum available $M$ will be a 1-row
matrix with 5 columns where

  * Column 1: total number of records
  * Column 2: total number of events
  * Column 3: median of $\hat{S}$
  * Column 4: lower bound of $100(1-\alpha)\%$ confidence interval for
    the median of $\hat{S}$
  * Column 5: upper bound of $100(1-\alpha)\%$ confidence interval for
    the median of $\hat{S}$.

If a comparison of the survival data across multiple groups needs to be
performed, `KM.dml` computes two matrices $T$ and
$$T\_GROUPS\_OE$$ that contain a summary of the test. The 1-row matrix $T$
stores the following statistics:

  * Column 1: number of groups in the survival data
  * Column 2: degree of freedom for Chi-squared distributed test
    statistic
  * Column 3: value of test statistic
  * Column 4: $P$-value.

Matrix $$T\_GROUPS\_OE$$ contains the following statistics for each of $g$
groups:

  * Column 1: number of events
  * Column 2: number of observed death times ($O$)
  * Column 3: number of expected death times ($E$)
  * Column 4: $(O-E)^2/E$
  * Column 5: $(O-E)^2/V$.


* * *

## 6.2. Cox Proportional Hazard Regression Model

### Description

The Cox (proportional hazard or PH) is a semi-parametric statistical
approach commonly used for analyzing survival data. Unlike
non-parametric approaches, e.g., the [Kaplan-Meier estimates](algorithms-survival-analysis.html#kaplan-meier-survival-analysis),
which can be used to analyze single sample of
survival data or to compare between groups of survival times, the Cox PH
models the dependency of the survival times on the values of
*explanatory variables* (i.e., covariates) recorded for
each individual at the time origin. Our focus is on covariates that do
not change value over time, i.e., time-independent covariates, and that
may be categorical (ordinal or nominal) as well as continuous-valued.


### Usage

**Cox**:

    hadoop jar SystemML.jar -f Cox.dml
                            -nvargs X=<file>
                                    TE=<file>
                                    F=<file>
                                    R=[file]
                                    M=<file>
                                    S=[file]
                                    T=[file]
                                    COV=<file>
                                    RT=<file>
                                    XO=<file>
                                    MF=<file>
                                    alpha=[double]
                                    tol=[double]
                                    moi=[int]
                                    mii=[int]
                                    fmt=[format]

**Cox Prediction**:

    hadoop jar SystemML.jar -f Cox-predict.dml
                            -nvargs X=<file>
                                    RT=<file>
                                    M=<file>
                                    Y=<file>
                                    COV=<file>
                                    MF=<file>
                                    P=<file>
                                    fmt=[format]

### Arguments - Cox Model Fitting/Prediction

**X**: Location (on HDFS) to read the input matrix of the survival data
containing:

  * timestamps
  * whether event occurred (1) or data is censored (0)
  * feature vectors

**Y**: Location (on HDFS) to the read matrix used for prediction

**TE**: Location (on HDFS) to read the 1-column matrix $TE$ that contains the
column indices of the input matrix $X$ corresponding to timestamps
(first entry) and event information (second entry)

**F**: Location (on HDFS) to read the 1-column matrix $F$ that contains the
column indices of the input matrix $X$ corresponding to the features to
be used for fitting the Cox model

**R**: (default: `" "`) If factors (i.e., categorical features) are available in the
input matrix $X$, location (on HDFS) to read matrix $R$ containing the
start (first column) and end (second column) indices of each factor in
$X$; alternatively, user can specify the indices of the baseline level
of each factor which needs to be removed from $X$. If $R$ is not
provided by default all variables are considered to be
continuous-valued.

**M**: Location (on HDFS) to store the results of Cox regression analysis
including regression coefficients $\beta_j$s, their standard errors,
confidence intervals, and $P$-values

**S**: (default: `" "`) Location (on HDFS) to store a summary of some statistics of
the fitted model including number of records, number of events,
log-likelihood, AIC, Rsquare (Cox & Snell), and maximum possible Rsquare

**T**: (default: `" "`) Location (on HDFS) to store the results of Likelihood ratio
test, Wald test, and Score (log-rank) test of the fitted model

**COV**: Location (on HDFS) to store the variance-covariance matrix of
$\beta_j$s; note that parameter `COV` needs to be provided as
input to prediction.

**RT**: Location (on HDFS) to store matrix $RT$ containing the order-preserving
recoded timestamps from $X$; note that parameter `RT` needs
to be provided as input for prediction.

**XO**: Location (on HDFS) to store the input matrix $X$ ordered by the
timestamps; note that parameter `XO` needs to be provided as
input for prediction.

**MF**: Location (on HDFS) to store column indices of $X$ excluding the baseline
factors if available; note that parameter `MF` needs to be
provided as input for prediction.

**P**: Location (on HDFS) to store matrix $P$ containing the results of
prediction

**alpha**: (default: `0.05`) Parameter to compute a $100(1-\alpha)\%$ confidence interval
for $\beta_j$s

**tol**: (default: `0.000001`) Tolerance ($\epsilon$) used in the convergence criterion

**moi**: (default: `100`) Maximum number of outer (Fisher scoring) iterations

**mii**: (default: `0`) Maximum number of inner (conjugate gradient) iterations, or 0
if no maximum limit provided

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.


### Examples

**Cox**:

    hadoop jar SystemML.jar -f Cox.dml
                            -nvargs X=/user/ml/X.mtx
                                    TE=/user/ml/TE
                                    F=/user/ml/F
                                    R=/user/ml/R
                                    M=/user/ml/model.csv
                                    T=/user/ml/test.csv
                                    COV=/user/ml/var-covar.csv
                                    XO=/user/ml/X-sorted.mtx
                                    fmt=csv

    hadoop jar SystemML.jar -f Cox.dml
                            -nvargs X=/user/ml/X.mtx
                                    TE=/user/ml/TE
                                    F=/user/ml/F
                                    R=/user/ml/R
                                    M=/user/ml/model.csv
                                    T=/user/ml/test.csv
                                    COV=/user/ml/var-covar.csv
                                    RT=/user/ml/recoded-timestamps.csv
                                    XO=/user/ml/X-sorted.csv
                                    MF=/user/ml/baseline.csv
                                    alpha=0.01
                                    tol=0.000001
                                    moi=100
                                    mii=20
                                    fmt=csv

**Cox Prediction**:

    hadoop jar SystemML.jar -f Cox-predict.dml
                            -nvargs X=/user/ml/X-sorted.mtx
                                    RT=/user/ml/recoded-timestamps.csv
                                    M=/user/ml/model.csv
                                    Y=/user/ml/Y.mtx
                                    COV=/user/ml/var-covar.csv
                                    MF=/user/ml/baseline.csv
                                    P=/user/ml/predictions.csv
                                    fmt=csv


### Details

In the Cox PH regression model, the relationship between the hazard
function — i.e., the probability of event occurrence at a given time — and
the covariates is described as

$$
\begin{equation}
h_i(t)=h_0(t)\exp\Bigl\{ \sum_{j=1}^{p} \beta_jx_{ij} \Bigr\}
\end{equation}
$$

where the hazard function for the $i$th individual
($$i\in\{1,2,\ldots,n\}$$) depends on a set of $p$ covariates
$$x_i=(x_{i1},x_{i2},\ldots,x_{ip})$$, whose importance is measured by the
magnitude of the corresponding coefficients
$$\beta=(\beta_1,\beta_2,\ldots,\beta_p)$$. The term $$h_0(t)$$ is the
baseline hazard and is related to a hazard value if all covariates equal
0. In the Cox PH model the hazard function for the individuals may vary
over time, however the baseline hazard is estimated non-parametrically
and can take any form. Note that re-writing (1) we have

$$\log\biggl\{ \frac{h_i(t)}{h_0(t)} \biggr\} = \sum_{j=1}^{p} \beta_jx_{ij}$$

Thus, the Cox PH model is essentially a linear model for the logarithm
of the hazard ratio and the hazard of event for any individual is a
constant multiple of the hazard of any other. We follow similar notation
and methodology as in Section 3 of 
[[Collett2003]](algorithms-bibliography.html). For
completeness we briefly discuss the equations used in our
implementation.

**Factors in the model.** Note that if some of the feature variables are
factors they need to *dummy code* as follows. Let $\alpha$
be such a variable (i.e., a factor) with $a$ levels. We introduce $a-1$
indicator (or dummy coded) variables $X_2,X_3\ldots,X_a$ with $X_j=1$ if
$\alpha=j$ and 0 otherwise, for $$j\in\{ 2,3,\ldots,a\}$$. In particular,
one of $a$ levels of $\alpha$ will be considered as the baseline and is
not included in the model. In our implementation, user can specify a
baseline level for each of the factor (as selecting the baseline level
for each factor is arbitrary). On the other hand, if for a given factor
$\alpha$ no baseline is specified by the user, the most frequent level
of $\alpha$ will be considered as the baseline.

**Fitting the model.** We estimate the coefficients of the Cox model via
negative log-likelihood method. In particular the Cox PH model is fitted
by using trust region Newton method with conjugate
gradient [[Nocedal2006]](algorithms-bibliography.html). Define the risk set $R(t_j)$ at time
$t_j$ to be the set of individuals who die at time $t_i$ or later. The
PH model assumes that survival times are distinct. In order to handle
tied observations we use the *Breslow* approximation of the likelihood
function

$$\mathcal{L}=\prod_{j=1}^{r} \frac{\exp(\beta^\top s_j)}{\biggl\{ \sum_{l\in R(t_j)} \exp(\beta^\top x_l) \biggr\}^{d_j} }$$

where $d_j$ is number individuals who die at time $t_j$ and $s_j$
denotes the element-wise sum of the covariates for those individuals who
die at time $t_j$, $j=1,2,\ldots,r$, i.e., the $h$th element of $s_j$ is
given by $$s_{hj}=\sum_{k=1}^{d_j}x_{hjk}$$, where $x_{hjk}$ is the value
of $h$th variable ($$h\in \{1,2,\ldots,p\}$$) for the $k$th of the $d_j$
individuals ($$k\in\{ 1,2,\ldots,d_j \}$$) who die at the $j$th death time
($$j\in\{ 1,2,\ldots,r \}$$).

**Standard error and confidence interval for coefficients.** Note that
the variance-covariance matrix of the estimated coefficients
$\hat{\beta}$ can be approximated by the inverse of the Hessian
evaluated at $\hat{\beta}$. The square root of the diagonal elements of
this matrix are the standard errors of estimated coefficients. Once the
standard errors of the coefficients $se(\hat{\beta})$ is obtained we can
compute a $100(1-\alpha)\%$ confidence interval using
$$\hat{\beta}\pm z_{\alpha/2}se(\hat{\beta})$$, where $z_{\alpha/2}$ is
the upper $\alpha/2$-point of the standard normal distribution. In
`Cox.dml`, we utilize the built-in function
`inv()` to compute the inverse of the Hessian. Note that this
build-in function can be used only if the Hessian fits in the main
memory of a single machine.

**Wald test, likelihood ratio test, and log-rank test.** In order to
test the *null hypothesis* that all of the coefficients
$\beta_j$s are 0, our implementation provides three statistical test:
*Wald test*, *likelihood ratio test*, the
*log-rank test* (also known as the *score
test*). Let $p$ be the number of coefficients. The Wald test is
based on the test statistic ${\hat{\beta}}^2/{se(\hat{\beta})}^2$, which
is compared to percentage points of the Chi-squared distribution to
obtain the $P$-value. The likelihood ratio test relies on the test
statistic $$-2\log\{ {L}(\textbf{0})/{L}(\hat{\beta}) \}$$ ($\textbf{0}$
denotes a zero vector of size $p$ ) which has an approximate Chi-squared
distribution with $p$ degrees of freedom under the null hypothesis that
all $\beta_j$s are 0. The Log-rank test is based on the test statistic
$l=\nabla^\top L(\textbf{0}) {\mathcal{H}}^{-1}(\textbf{0}) \nabla L(\textbf{0})$,
where $\nabla L(\textbf{0})$ is the gradient of $L$ and
$\mathcal{H}(\textbf{0})$ is the Hessian of $L$ evaluated at **0**.
Under the null hypothesis that $\beta=\textbf{0}$, $l$ has a Chi-squared
distribution on $p$ degrees of freedom.

**Prediction.** Once the parameters of the model are fitted, we compute
the following predictions together with their standard errors

  * linear predictors
  * risk
  * estimated cumulative hazard

Given feature vector $$X_i$$ for individual $$i$$, we obtain the above
predictions at time $$t$$ as follows. The linear predictors (denoted as
$$\mathcal{LP}$$) as well as the risk (denoted as $\mathcal{R}$) are
computed relative to a baseline whose feature values are the mean of the
values in the corresponding features. Let $$X_i^\text{rel} = X_i - \mu$$,
where $$\mu$$ is a row vector that contains the mean values for each
feature. We have $$\mathcal{LP}=X_i^\text{rel} \hat{\beta}$$ and
$$\mathcal{R}=\exp\{ X_i^\text{rel}\hat{\beta} \}$$. The standard errors
of the linear predictors $$se\{\mathcal{LP} \}$$ are computed as the
square root of $${(X_i^\text{rel})}^\top V(\hat{\beta}) X_i^\text{rel}$$
and the standard error of the risk $$se\{ \mathcal{R} \}$$ are given by
the square root of
$${(X_i^\text{rel} \odot \mathcal{R})}^\top V(\hat{\beta}) (X_i^\text{rel} \odot \mathcal{R})$$,
where $$V(\hat{\beta})$$ is the variance-covariance matrix of the
coefficients and $$\odot$$ is the element-wise multiplication.

We estimate the cumulative hazard function for individual $i$ by

$$\hat{H}_i(t) = \exp(\hat{\beta}^\top X_i) \hat{H}_0(t)$$

where
$$\hat{H}_0(t)$$ is the *Breslow estimate* of the cumulative baseline
hazard given by

$$\hat{H}_0(t) = \sum_{j=1}^{k} \frac{d_j}{\sum_{l\in R(t_{(j)})} \exp(\hat{\beta}^\top X_l)}$$

In the equation above, as before, $d_j$ is the number of deaths, and
$$R(t_{(j)})$$ is the risk set at time $$t_{(j)}$$, for
$$t_{(k)} \leq t \leq t_{(k+1)}$$, $$k=1,2,\ldots,r-1$$. The standard error
of $$\hat{H}_i(t)$$ is obtained using the estimation

$$se\{ \hat{H}_i(t) \} = \sum_{j=1}^{k} \frac{d_j}{ {\left[ \sum_{l\in R(t_{(j)})} \exp(X_l\hat{\beta}) \right]}^2 } + J_i^\top(t) V(\hat{\beta}) J_i(t)\$$

where

$$J_i(t) = \sum_{j-1}^{k} d_j \frac{\sum_{l\in R(t_{(j)})} (X_l-X_i)\exp \{ (X_l-X_i)\hat{\beta} \}}{ {\left[ \sum_{l\in R(t_{(j)})} \exp\{(X_l-X_i)\hat{\beta}\} \right]}^2  }$$

for $$t_{(k)} \leq t \leq t_{(k+1)}$, $k=1,2,\ldots,r-1$$.


### Returns

Below we list the results of fitting a Cox regression model stored in
matrix $M$ with the following schema:

  * Column 1: estimated regression coefficients $\hat{\beta}$
  * Column 2: $\exp(\hat{\beta})$
  * Column 3: standard error of the estimated coefficients
    $se\{\hat{\beta}\}$
  * Column 4: ratio of $\hat{\beta}$ to $se\{\hat{\beta}\}$ denoted by
    $Z$
  * Column 5: $P$-value of $Z$
  * Column 6: lower bound of $100(1-\alpha)\%$ confidence interval for
    $\hat{\beta}$
  * Column 7: upper bound of $100(1-\alpha)\%$ confidence interval for
    $\hat{\beta}$.

Note that above $Z$ is the Wald test statistic which is asymptotically
standard normal under the hypothesis that $\beta=\textbf{0}$.

Moreover, `Cox.dml` outputs two log files `S` and
`T` containing a summary statistics of the fitted model as
follows. File `S` stores the following information

  * Line 1: total number of observations
  * Line 2: total number of events
  * Line 3: log-likelihood (of the fitted model)
  * Line 4: AIC
  * Line 5: Cox & Snell Rsquare
  * Line 6: maximum possible Rsquare.

Above, the AIC is computed as in [[Stepwise Linear Regression]](algorithms-regression.html#stepwise-linear-regression), the Cox & Snell Rsquare
is equal to $$1-\exp\{ -l/n \}$$, where $l$ is the log-rank test statistic
as discussed above and $n$ is total number of observations, and the
maximum possible Rsquare computed as $$1-\exp\{ -2 L(\textbf{0})/n \}$$,
where $L(\textbf{0})$ denotes the initial likelihood.

File `T` contains the following information

  * Line 1: Likelihood ratio test statistic, degree of freedom of the
    corresponding Chi-squared distribution, $P$-value
  * Line 2: Wald test statistic, degree of freedom of the corresponding
    Chi-squared distribution, $P$-value
  * Line 3: Score (log-rank) test statistic, degree of freedom of the
    corresponding Chi-squared distribution, $P$-value.

Additionally, the following matrices will be stored. Note that these
matrices are required for prediction.

  * Order-preserving recoded timestamps $RT$, i.e., contiguously
    numbered from 1 $\ldots$ \#timestamps
  * Feature matrix ordered by the timestamps $XO$
  * Variance-covariance matrix of the coefficients $COV$
  * Column indices of the feature matrix with baseline factors removed
    (if available) $MF$.

**Prediction.** Finally, the results of prediction is stored in Matrix
$P$ with the following schema

  * Column 1: linear predictors
  * Column 2: standard error of the linear predictors
  * Column 3: risk
  * Column 4: standard error of the risk
  * Column 5: estimated cumulative hazard
  * Column 6: standard error of the estimated cumulative hazard.


