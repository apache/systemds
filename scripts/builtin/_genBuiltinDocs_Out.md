
Markdown Files for scripts
==========================

# Overview

## abstain-Function


This function calls the multiLogReg-function in which solves Multinomial Logistic Regression using Trust Region method
### Usage


```python
abstain(X, Y, threshold, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Location to read the matrix of feature vectors|
|Y|Matrix[Double]|---|Location to read the matrix with category labels|
|threshold|Double|0.0|---|
|verbose|Boolean|FALSE|flag specifying if logging information should be printed|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|abstain|Matrix[Double]|---|

## als-Function


This script computes an approximate factorization of a low-rank matrix X into two matrices U and V

using different implementations of the Alternating-Least-Squares (ALS) algorithm.

Matrices U and V are computed by minimizing a loss function (with regularization).
### Usage


```python
als(X, rank, reg, "L2", f, +, "wL2", f, +, +, lambda, maxi, check, thr, if, if)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|String|---|Location to read the input matrix X to be factorized|
|rank|Int|10|Rank of the factorization|
|reg|String|"L2"|Regularization:|
|"L2"|=|L2|regularization;|
|f|(U,|V)|= 0.5 * sum (W * (U %*% V - X) ^ 2)|
|+|0.5|*|lambda * (sum (U ^ 2) + sum (V ^ 2))|
|"wL2"|=|weighted|L2 regularization|
|f|(U,|V)|= 0.5 * sum (W * (U %*% V - X) ^ 2)|
|+|0.5|*|lambda * (sum (U ^ 2 * row_nonzeros)|
|+|sum|(V|^ 2 * col_nonzeros))|
|lambda|Double|0.000001|Regularization parameter, no regularization if 0.0|
|maxi|Int|50|Maximum number of iterations|
|check|Boolean|TRUE|Check for convergence after every iteration, i.e., updating U and V once|
|thr|Double|0.0001|Assuming check is set to TRUE, the algorithm stops and convergence is declared|
|if|the|decrease|in loss in any two consecutive iterations falls below this threshold;|
|if|check|is|FALSE thr is ignored|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|U|Matrix|An m x r matrix where r is the factorization rank|
|V|Matrix|An m x r matrix where r is the factorization rank|

## alsCG-Function


This script computes an approximate factorization of a low-rank matrix X into two matrices U and V

using the Alternating-Least-Squares (ALS) algorithm with conjugate gradient.

Matrices U and V are computed by minimizing a loss function (with regularization).
### Usage


```python
alsCG(X, rank, reg, "L2", f, +, "wL2", f, +, +, lambda, maxi, check, thr, if, if)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|String|---|Location to read the input matrix X to be factorized|
|rank|Int|10|Rank of the factorization|
|reg|String|"L2"|Regularization:|
|"L2"|=|L2|regularization;|
|f|(U,|V)|= 0.5 * sum (W * (U %*% V - X) ^ 2)|
|+|0.5|*|lambda * (sum (U ^ 2) + sum (V ^ 2))|
|"wL2"|=|weighted|L2 regularization|
|f|(U,|V)|= 0.5 * sum (W * (U %*% V - X) ^ 2)|
|+|0.5|*|lambda * (sum (U ^ 2 * row_nonzeros)|
|+|sum|(V|^ 2 * col_nonzeros))|
|lambda|Double|0.000001|Regularization parameter, no regularization if 0.0|
|maxi|Int|50|Maximum number of iterations|
|check|Boolean|TRUE|Check for convergence after every iteration, i.e., updating U and V once|
|thr|Double|0.0001|Assuming check is set to TRUE, the algorithm stops and convergence is declared|
|if|the|decrease|in loss in any two consecutive iterations falls below this threshold;|
|if|check|is|FALSE thr is ignored|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|U|Matrix|An m x r matrix where r is the factorization rank|
|V|Matrix|An m x r matrix where r is the factorization rank|

## alsDS-Function


Alternating-Least-Squares (ALS) algorithm using a direct solve method for

individual least squares problems (reg="L2"). This script computes an

approximate factorization of a low-rank matrix V into two matrices L and R.

Matrices L and R are computed by minimizing a loss function (with regularization).
### Usage


```python
alsDS(V, L, R, rank, lambda, maxi, check, thr, if, if)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|V|String|---|Location to read the input matrix V to be factorized|
|L|String|---|Location to write the factor matrix L|
|R|String|---|Location to write the factor matrix R|
|rank|Int|10|Rank of the factorization|
|lambda|Double|0.000001|Regularization parameter, no regularization if 0.0|
|maxi|Int|50|Maximum number of iterations|
|check|Boolean|FALSE|Check for convergence after every iteration, i.e., updating L and R once|
|thr|Double|0.0001|Assuming check is set to TRUE, the algorithm stops and convergence is declared|
|if|the|decrease|in loss in any two consecutive iterations falls below this threshold;|
|if|check|is|FALSE thr is ignored|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|U|Matrix|An m x r matrix where r is the factorization rank|
|V|Matrix|An m x r matrix where r is the factorization rank|

## alsPredict-Function


This script computes the rating/scores for a given list of userIDs

using 2 factor matrices L and R. We assume that all users have rates

at least once and all items have been rates at least once.
### Usage


```python
alsPredict(userIDs, I, L, R)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|userIDs|Matrix|---|Column vector of user-ids (n x 1)|
|I|Matrix|---|Indicator matrix user-id x user-id to exclude from scoring|
|L|Matrix|---|The factor matrix L: user-id x feature-id|
|R|Matrix|---|The factor matrix R: feature-id x item-id|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix|The output user-id/item-id/score#|

## alsTopkPredict-Function


This script computes the top-K rating/scores for a given list of userIDs

using 2 factor matrices L and R. We assume that all users have rates

at least once and all items have been rates at least once.
### Usage


```python
alsTopkPredict(userIDs, I, L, R, K)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|userIDs|Matrix|---|Column vector of user-ids (n x 1)|
|I|Matrix|---|Indicator matrix user-id x user-id to exclude from scoring|
|L|Matrix|---|The factor matrix L: user-id x feature-id|
|R|Matrix|---|The factor matrix R: feature-id x item-id|
|K|Int|5|The number of top-K items|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|TopIxs|Matrix|A matrix containing the top-K item-ids with highest predicted ratings for the specified users (rows)|
|TopVals|Matrix|A matrix containing the top-K predicted ratings for the specified users (rows)|

## applyAndEvaluate-Function


This script will read the dirty and clean data, then it will apply the best pipeline on dirty data

and then will classify both cleaned dataset and check if the cleaned dataset is performing same as original dataset

in terms of classification accuracy
### Usage


```python
applyAndEvaluate(trainData, testData, metaData, lp, pip, hp, evaluationFunc, evalFunHp, isLastLabel, correctTypos)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|trainData|Frame[Unknown]|---||
|testData|Frame[Unknown]|---||
|metaData|Frame[Unknown]|as.frame("NULL")||
|lp|Frame[Unknown]|---||
|pip|Frame[Unknown]|---||
|hp|Frame[Unknown]|---||
|evaluationFunc|String|---||
|evalFunHp|Matrix[Double]|---||
|isLastLabel|Boolean|TRUE||
|correctTypos|Boolean|FALSE||

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|result|Matrix[Double]|---|

## arima-Function


Builtin function that implements ARIMA
### Usage


```python
arima(X, max_func_invoc, p, d, q, P, D, Q, s, include_mean, solver)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|---|The input Matrix to apply Arima on.|
|max_func_invoc|Int|1000||
|p|Int|0|non-seasonal AR order|
|d|Int|0|non-seasonal differencing order|
|q|Int|0|non-seasonal MA order|
|P|Int|0|seasonal AR order|
|D|Int|0|seasonal differencing order|
|Q|Int|0|seasonal MA order|
|s|Int|1|period in terms of number of time-steps|
|include_mean|Boolean|FALSE|center to mean 0, and include in result|
|solver|String|jacobi|solver, is either "cg" or "jacobi"|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|best_point|String|The calculated coefficients|

## autoencoder_2layer-Function


Trains a 2-layer autoencoder with minibatch SGD and step-size decay.

If invoked with H1 > H2 then it becomes a 'bowtie' structured autoencoder

Weights are initialized using Glorot & Bengio (2010) AISTATS initialization.

The script standardizes the input before training (can be turned off).

Also, it randomly reshuffles rows before training.

Currently, tanh is set to be the activation function.

By re-implementing 'func' DML-bodied function, one can change the activation.
### Usage


```python
autoencoder_2layer(X, H1, H2, EPOCH, OBJ, at, objective, BATCH, STEP, DECAY, MOMENTUM)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|String|---|Filename where the input is stored|
|H1|Int|---|Number of neurons in the 1st hidden layer|
|H2|Int|---|Number of neurons in the 2nd hidden layer|
|EPOCH|Int|---|Number of epochs to train for|
|OBJ|Boolean|FALSE|If TRUE, Computes objective function value (squared-loss)|
|at|the|end|of each epoch. Note that, computing the full|
|objective|can|take|a lot of time.|
|BATCH|Int|256|Mini-batch size (training parameter)|
|STEP|Double|1e-5|Initial step size (training parameter)|
|DECAY|Double|0.95|Decays step size after each epoch (training parameter)|
|MOMENTUM|Double|0.9|Momentum parameter (training parameter)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|W1_out|Matrix[Double]|Matrix storing weights between input layer and 1st hidden layer|
|b1_out|Matrix[Double]|Matrix storing bias between input layer and 1st hidden layer|
|W2_out|Matrix[Double]|Matrix storing weights between 1st hidden layer and 2nd hidden layer|
|b2_out|Matrix[Double]|Matrix storing bias between 1st hidden layer and 2nd hidden layer|
|W3_out|Matrix[Double]|Matrix storing weights between 2nd hidden layer and 3rd hidden layer|
|b3_out|Matrix[Double]|Matrix storing bias between 2nd hidden layer and 3rd hidden layer|
|W4_out|Matrix[Double]|Matrix storing weights between 3rd hidden layer and output layer|
|b4_out|Matrix[Double]|Matrix storing bias between 3rd hidden layer and output layer|
|HIDDEN|Matrix[Double]|Matrix storing the hidden (2nd) layer representation if needed|

## bandit-Function


in The bandit function the objective is to find an arm that optimises a known functional of the unknown arm-reward 
distributions.
### Usage


```python
bandit(X_train, Y_train, X_test, Y_test, metaList, evaluationFunc, evalFunHp, lp, primitives, params, K, R, cvk, verbose, output)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X_train|Matrix[Double]|---||
|Y_train|Matrix[Double]|---||
|X_test|Matrix[Double]|---||
|Y_test|Matrix[Double]|---||
|metaList|List[Unknown]|---||
|evaluationFunc|String|---||
|evalFunHp|Matrix[Double]|---||
|lp|Frame[Unknown]|---||
|primitives|Frame[Unknown]|---||
|params|Frame[Unknown]|---||
|K|Integer|3||
|R|Integer|50||
|cvk|Integer|2||
|verbose|Boolean|TRUE||
|output|String|""||

### Returns

|OUTPUT:|
| :---: |
|NAME      TYPE             MEANING|

## bivar-Function


For a given pair of attribute sets, compute bivariate statistics between all attribute pairs.

Given, index1 = {A_11, A_12, ... A_1m} and index2 = {A_21, A_22, ... A_2n}

compute bivariate stats for m*n pairs (A_1i, A_2j), (1<= i <=m) and (1<= j <=n).
### Usage


```python
bivar(X, S1, S2, T1, (kind=1, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input matrix|
|S1|Matrix[Integer]|---|First attribute set {A_11, A_12, ... A_1m}|
|S2|Matrix[Integer]|---|Second attribute set {A_21, A_22, ... A_2n}|
|T1|Matrix[Integer]|---|Kind for attributes in S1|
|(kind=1|for|scale,|kind=2 for nominal, kind=3 for ordinal)|
|verbose|Boolean|---|Print bivar stats|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|basestats_scale_scale|Matrix|basestats_scale_scale as output with bivar stats|
|basestats_nominal_scale|Matrix|basestats_nominal_scale as output with bivar stats|
|basestats_nominal_nominal|Matrix|basestats_nominal_nominal as output with bivar stats|
|basestats_ordinal_ordinal|Matrix|basestats_ordinal_ordinal as output with bivar stats|

## components-Function


Computes the connected components of a graph and returns a

vector indicating the assignment of vertices to components,

where each component is identified by the maximum vertex ID
### Usage


```python
components(X, Y, icpt, no, 2, tol, reg, maxi, maxii, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix|---|Location to read the matrix of feature vectors|
|Y|Matrix|---|Location to read the matrix with category labels|
|icpt|Integer|0|Intercept presence, shifting and rescaling X columns: 0 = no intercept,|
|no|shifting,|no|rescaling; 1 = add intercept, but neither shift nor rescale X;|
|2|=|add|intercept, shift & rescale X columns to mean = 0, variance = 1|
|tol|Double|0.000001|tolerance ("epsilon")|
|reg|Double|0.0|regularization parameter (lambda = 1/C); intercept is not regularized|
|maxi|Integer|100|max. number of outer (Newton) iterations|
|maxii|Integer|0|max. number of inner (conjugate gradient) iterations, 0 = no max|
|verbose|Boolean|FALSE|flag specifying if logging information should be printed|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|betas|Matrix[Double]|regression betas as output for prediction|

## confusionMatrix-Function


Accepts a vector for prediction and a one-hot-encoded matrix

Then it computes the max value of each vector and compare them

After which, it calculates and returns the sum of classifications

and the average of each true class.
### Usage


```python
confusionMatrix(P, Y, encoded)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|P|Double|---|vector of Predictions|
|Y|Double|---|vector of Golden standard One Hot Encoded; the one hot|
|encoded|vector|of|actual labels|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|ConfusionSum|Double|The Confusion Matrix Sums of classifications|
|ConfusionAvg|Double|The Confusion Matrix averages of each true class|

## cor-Function


This Function compute correlation matrix in vectorized form
### Usage


```python
cor(X)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|---||

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Double|Correlation matrix|

## correctTypos-Function


References:

Fred J. Damerau. 1964.
### Usage


```python
correctTypos(strings, nullMask, frequency_threshold, distance_threshold, decapitalize, correct, is_verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|strings|Frame[String]|---|The nx1 input frame of corrupted strings|
|nullMask|Matrix[Double]|---|---|
|frequency_threshold|Double|0.05|Strings that occur above this frequency level will not be corrected|
|distance_threshold|integer|2|Max distance at which strings are considered similar|
|decapitalize|Boolean|TRUE|Decapitalize all strings before correction|
|correct|Boolean|TRUE|Correct strings or only report potential errors|
|is_verbose|Boolean|FALSE|Print debug information|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Frame[String]|Corrected nx1 output frame|

## cox-Function


This script fits a cox Proportional hazard regression model.

The Breslow method is used for handling ties and the regression parameters

are computed using trust region newton method with conjugate gradient
### Usage


```python
cox(X, containing, 2:, 3:, TE, (first, F, fitting, R, X,, the, R[,1]:, R[,2]:, Alternatively,, each, and, if, alpha, tol, moi, mii)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix|---|Location to read the input matrix X containing the survival data|
|containing|the|following|information|
|2:|whether|an|event occurred (1) or data is censored (0)|
|3:|feature|vectors||
|TE|Matrix|---|Column indices of X as a column vector which contain timestamp|
|(first|row)|and|event information (second row)|
|F|Matrix|---|Column indices of X as a column vector which are to be used for|
|fitting|the|Cox|model|
|R|Matrix|---|If factors (categorical variables) are available in the input matrix|
|X,|location|to|read matrix R containing the start and end indices of|
|the|factors|in|X|
|R[,1]:|start|indices||
|R[,2]:|end|indices||
|Alternatively,|user|can|specify the indices of the baseline level of|
|each|factor|which|needs to be removed from X; in this case the start|
|and|end|indices|corresponding to the baseline level need to be the same;|
|if|R|is|not provided by default all variables are considered to be continuous|
|alpha|Double|0.05|Parameter to compute a 100*(1-alpha)% confidence interval for the betas|
|tol|Double|0.000001|Tolerance ("epsilon")|
|moi|Int|100|Max. number of outer (Newton) iterations|
|mii|Int|0|Max. number of inner (conjugate gradient) iterations, 0 = no max|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Matrix[Double]|A D x 7 matrix M, where D denotes the number of covariates, with the following schema:|
|M[,1]:|betas||
|M[,2]:|exp(betas)||
|M[,3]:|standard|error of betas|
|M[,4]:|Z||
|M[,5]:|P-value||
|M[,6]:|lower|100*(1-alpha)% confidence interval of betas|
|M[,7]:|upper|100*(1-alpha)% confidence interval of betas|
|S,T|Matrix[Double]|Two matrices containing a summary of some statistics of the fitted model:|
|1|-|File S with the following format|
|-|row|1: no. of observations|
|-|row|2: no. of events|
|-|row|3: log-likelihood|
|-|row|4: AIC|
|-|row|5: Rsquare (Cox & Snell)|
|-|row|6: max possible Rsquare|
|2|-|File T with the following format|
|-|row|1: Likelihood ratio test statistic, degree of freedom, P-value|
|-|row|2: Wald test statistic, degree of freedom, P-value|
|-|row|3: Score (log-rank) test statistic, degree of freedom, P-value|
|RT,XO,COV|Matrix[Double]|Additionally, the following matrices are stored (needed for prediction)|
|1-|A|column matrix RT that contains the order-preserving recoded timestamps from X|
|2-|Matrix|XO which is matrix X with sorted timestamps|
|3-|Variance-covariance|matrix of the betas COV|
|4-|A|column matrix MF that contains the column indices of X with the baseline factors removed (if available)|

## cspline-Function


THIS SCRIPT SOLVES CUBIC SPLINE INTERPOLATION
### Usage


```python
cspline(X, monotonically, Y, inp_x, mode, tol, L2, maxi)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|1-column matrix of x values knots. It is assumed that x values are|
|monotonically|increasing|and|there is no duplicates points in X|
|Y|Matrix[Double]|---|1-column matrix of corresponding y values knots|
|inp_x|Double|---|the given input x, for which the cspline will find predicted y|
|mode|String|"DS"|Specifies the method for cspline (DS - Direct Solve, CG - Conjugate Gradient)|
|tol|Double|-1.0|Tolerance (epsilon); conjugate graduent procedure terminates early if|
|L2|norm|of|the beta-residual is less than tolerance * its initial norm|
|maxi|Integer|-1|Maximum number of conjugate gradient iterations, 0 = no maximum|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|pred_Y|Matrix[Double]|Predicted value|
|K|Matrix[Double]|Matrix of k parameters|

## csplineCG-Function


THIS SCRIPT SOLVES CUBIC SPLINE INTERPOLATION USING THE CONJUGATE GRADIENT ALGORITHM
### Usage


```python
csplineCG(X, monotonically, Y, inp_x, tol, L2, maxi)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|1-column matrix of x values knots. It is assumed that x values are|
|monotonically|increasing|and|there is no duplicates points in X|
|Y|Matrix[Double]|---|1-column matrix of corresponding y values knots|
|inp_x|Double|---|the given input x, for which the cspline will find predicted y.|
|tol|Double|0.000001|Tolerance (epsilon); conjugate graduent procedure terminates early if|
|L2|norm|of|the beta-residual is less than tolerance * its initial norm|
|maxi|Integer|0|Maximum number of conjugate gradient iterations, 0 = no maximum|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|pred_Y|Matrix[Double]|Predicted value|
|K|Matrix[Double]|Matrix of k parameters|

## csplineDS-Function


THIS SCRIPT SOLVES CUBIC SPLINE INTERPOLATION USING THE DIRECT SOLVER
### Usage


```python
csplineDS(X, monotonically, Y, inp_x)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|1-column matrix of x values knots. It is assumed that x values are|
|monotonically|increasing|and|there is no duplicates points in X|
|Y|Matrix[Double]|---|1-column matrix of corresponding y values knots|
|inp_x|Double|---|the given input x, for which the cspline will find predicted y.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|pred_y|Matrix[Double]|Predicted value|
|K|Matrix[Double]|Matrix of k parameters|

## cvlm-Function


The cvlm-function is used for cross-validation of the provided data model. This function follows a non-exhaustive cross

validation method. It uses lm and lmPredict functions to solve the linear regression and to predict the class of a

feature vector with no intercept, shifting, and rescaling.
### Usage


```python
cvlm(X, y, k, icpt, reg, highly)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Recorded Data set into matrix|
|y|Matrix[Double]|---|1-column matrix of response values.|
|k|Integer|---|Number of subsets needed, It should always be more than 1 and less than nrow(X)|
|icpt|Integer|0|Intercept presence, shifting and rescaling the columns of X|
|reg|Double|1e-7|Regularization constant (lambda) for L2-regularization. set to nonzero for|
|highly|dependant/sparse/numerous|features||

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|y_predict|Matrix[Double]|Response values|
|allbeta|Matrix[Double]|Validated data set|

## dbscan-Function


Implements the DBSCAN clustering algorithm using Euclidian distance matrix
### Usage


```python
dbscan(X, eps, minPts, (includes)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|The input Matrix to do DBSCAN on.|
|eps|Double|0.5|Maximum distance between two points for one to be considered reachable for the other.|
|minPts|Int|5|Number of points in a neighborhood for a point to be considered as a core point|
|(includes|the|point|itself).|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|clusterMembers|Matrix[Double]|clustering Matrix|

## decisionTree-Function


THIS SCRIPT IMPLEMENTS CLASSIFICATION TREES WITH BOTH SCALE AND CATEGORICAL FEATURES
### Usage


```python
decisionTree(X, Y, R, -, a, If, bins, depth, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Feature matrix X; note that X needs to be both recoded and dummy coded|
|Y|Matrix[Double]|---|Label matrix Y; note that Y needs to be both recoded and dummy coded|
|R|Matrix[Double]|"|"      Matrix R which for each feature in X contains the following information|
|-|R[1,]:|Row|Vector which indicates if feature vector is scalar or categorical. 1 indicates|
|a|scalar|feature|vector, other positive Integers indicate the number of categories|
|If|R|is|not provided by default all variables are assumed to be scale|
|bins|Integer|20|Number of equiheight bins per scale feature to choose thresholds|
|depth|Integer|25|Maximum depth of the learned tree|
|verbose|Boolean|FALSE|boolean specifying if the algorithm should print information while executing|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Matrix[Double]|Matrix M where each column corresponds to a node in the learned tree and each row|
|contains|the|following information:|
|M[1,j]:|id|of node j (in a complete binary tree)|
|M[2,j]:|Offset|(no. of columns) to left child of j if j is an internal node, otherwise 0|
|M[3,j]:|Feature|index of the feature (scale feature id if the feature is scale or|
|categorical|feature|id if the feature is categorical)|
|that|node|j looks at if j is an internal node, otherwise 0|
|M[4,j]:|Type|of the feature that node j looks at if j is an internal node: holds|
|the|same|information as R input vector|
|M[5,j]:|If|j is an internal node: 1 if the feature chosen for j is scale,|
|otherwise|the|size of the subset of values|
|stored|in|rows 6,7,... if j is categorical|
|If|j|is a leaf node: number of misclassified samples reaching at node j|
|M[6:,j]:|If|j is an internal node: Threshold the example's feature value is compared|
|to|is|stored at M[6,j] if the feature chosen for j is scale,|
|otherwise|if|the feature chosen for j is categorical rows 6,7,... depict the value subset chosen for j|
|If|j|is a leaf node 1 if j is impure and the number of samples at j > threshold, otherwise 0|

## deepWalk-Function


This script performs DeepWalk on a given graph (https://arxiv.org/pdf/1403.6652.pdf)
### Usage


```python
deepWalk(Graph, w, d, gamma, t, alpha, beta)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|Graph|Matrix|---|adjacency matrix of a graph (n x n)|
|w|Integer|---|window size|
|d|Integer|---|embedding size|
|gamma|Integer|---|walks per vertex|
|t|Integer|---|walk length|
|alpha|Double|0.025|learning rate|
|beta|Double|0.9|factor for decreasing learning rate|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Phi|Matrix[Double]|matrix of vertex/word representation (n x d)|

## denialConstraints-Function


This function considers some constraints indicating statements that can NOT happen in the data (denial constraints).
### Usage


```python
denialConstraints(dataFrame, to, Recommended, constraintsFrame, 1., 2., -, -, -, ie, then, 3., 4., 5., 6., 7., in, 8.)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|dataFrame|Frame|---|frame which columns represent the variables of the data and the rows correspond|
|to|different|tuples|or instances.|
|Recommended|to|have|a column indexing the instances from 1 to N (N=number of instances).|
|constraintsFrame|Frame|---|frame with fixed columns and each row representing one constraint.|
|1.|idx:|(double)|index of the constraint, from 1 to M (number of constraints)|
|2.|constraint.type:|(string)|The constraints can be of 3 different kinds:|
|-|variableCompare:|for|each instance, it will compare the values of two variables (with a relation <, > or =).|
|-|valueCompare:|for|each instance, it will compare a fixed value and a variable value (with a relation <, > or =).|
|-|instanceCompare:|for|every couple of instances, it will compare the relation between two variables,|
|ie|if|the|value of the variable 1 in instance 1 is lower/higher than the value of variable 1 in instance 2,|
|then|the|value|of of variable 2 in instance 2 can't be lower/higher than the value of variable 2 in instance 2.|
|3.|group.by:|(boolean)|if TRUE only one group of data (defined by a variable option) will be considered for the constraint.|
|4.|group.variable:|(string,|only if group.by TRUE) name of the variable (column in dataFrame) that will divide our data in groups.|
|5.|group.option:|(only|if group.by TRUE) option of the group.variable that defines the group to consider.|
|6.|variable1:|(string)|first variable to compare (name of column in dataFrame).|
|7.|relation:|(string)|can be < , > or = in the case of variableCompare and valueCompare, and < >, < < , > < or > >|
|in|the|case|of instanceCompare|
|8.|variable2:|(string)|second variable to compare (name of column in dataFrame) or fixed value for the case of valueCompare.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|WrongInstances|Matrix[double]|Matrix of 2 columns.|
|-|First|column shows the indexes of dataFrame that are wrong.|
|-|Second|column shows the index of the denial constraint that is fulfilled|
|If|there|are no wrong instances to show (0 constrains fulfilled) --> WrongInstances=matrix(0,1,2)|

## discoverFD-Function


Implements builtin for finding functional dependencies
### Usage


```python
discoverFD(X, Mask, threshold)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|--|Input Matrix X, encoded Matrix if data is categorical|
|Mask|Double|--|A row vector for interested features i.e. Mask =[1, 0, 1]|
|threshold|Double|--|threshold value in interval [0, 1] for robust FDs|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|FD|Double|---        matrix of functional dependencies|

## dmv-Function


The dmv-function is used to find disguised missing values utilising syntactical pattern recognition.
### Usage


```python
dmv(X, threshold, that, replace)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Frame[String]|---|Input Frame|
|threshold|Double|0.8|Threshold value in interval [0, 1] for dominant pattern per column (e.g., 0.8 means|
|that|80%|of|the entries per column must adhere this pattern to be dominant)|
|replace|String|"NA"|The string disguised missing values are replaced with|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Frame[String]|Frame X including detected disguised missing values|

## ema-Function


This function imputes values with exponential moving average (single, double or triple).
### Usage


```python
ema(X, search_iterations, used, mode, freq, alpha, beta, gamma)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Frame[Double]|---|Frame that contains timeseries data that needs to be imputed|
|search_iterations|Integer|--|Budget iterations for parameter optimisation,|
|used|if|parameters|weren't set|
|mode|String|---|Type of EMA method. Either "single", "double" or "triple"|
|freq|Double|---|Seasonality when using triple EMA.|
|alpha|Double|---|alpha- value for EMA|
|beta|Double|---|beta- value for EMA|
|gamma|Double|---|gamma- value for EMA|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Frame[Double]|Frame with EMA results|

## executePipeline-Function


This function execute pipeline.
### Usage


```python
executePipeline(logical, pipeline, X, Y, Xtest, Ytest, metaList, hyperParameters, hpForPruning, changesByOp, flagsCount, test, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|logical|Frame[String]|NULL|---|
|pipeline|Frame[String]|---|---|
|X|Matrix[Double]|---|---|
|Y|Matrix[Double]|---|---|
|Xtest|Matrix[Double]|---|---|
|Ytest|Matrix[Double]|---|---|
|metaList|List[Unknown]|---|---|
|hyperParameters|Matrix[Double]|---|---|
|hpForPruning|Matrix[Double]|0|---|
|changesByOp|Matrix[Double]|0|---|
|flagsCount|Integer|---|---|
|test|Boolean|FALSE|---|
|verbose|Boolean|---|---|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|---|
|Y|Matrix[Double]|---|
|Xtest|Matrix[Double]|---|
|Ytest|Matrix[Double]|---|
|t2|Double|---|
|hpForPruning|Matrix[Double]|---|
|changesByOp|Matrix[Double]|---|

## ffPredict-Function


This builtin function makes prediction given data and trained feedforward neural network model
### Usage


```python
ffPredict(Model, X, batch_size)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|Model|List[unknown]|---|Trained ff neural network model|
|X|Matrix[double]|---|Data used for making predictions|
|batch_size|Integer|128|Batch size|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|pred|Double|Predicted value|

## ffTrain-Function


This builtin function trains simple feed-forward neural network. The architecture of the

networks is: affine1 -> relu -> dropout -> affine2 -> configurable output activation function.

Hidden layer has 128 neurons. Dropout rate is 0.35. Input and ouptut sizes are inferred from X and Y.
### Usage


```python
ffTrain(X, Y, batch_size, epochs, learning_rate, out_activation, "sigmoid",, loss_fcn, "l1",, shuffle, validation_split, seed, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[double]|---|Training data|
|Y|Matrix[Double]|---|Labels/Target values|
|batch_size|Integer|64|Batch size|
|epochs|Integer|20|Number of epochs|
|learning_rate|Double|0.003|Learning rate|
|out_activation|String|---|User specified ouptut activation function. Possible values:|
|"sigmoid",|"relu",|"lrelu",|"tanh", "softmax", "logits" (no activation).|
|loss_fcn|String|---|User specified loss function. Possible values:|
|"l1",|"l2",|"log_loss",|"logcosh_loss", "cel" (cross-entropy loss).|
|shuffle|Boolean|FALSE|Flag which indicates if dataset should be shuffled or not|
|validation_split|Double|0.0|Fraction of training set used as validation set|
|seed|Integer|-1|Seed for model initialization|
|verbose|Boolean|FALSE|Flag which indicates if function should print to stdout|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|model|List[unknown]|Trained model which can be used in ffPredict|

## frameSort-Function


Related to [SYSTEMDS-2662] dependency function for cleaning pipelines

Built-in for sorting frames
### Usage


```python
frameSort(F)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|F|Frame[String]|---|Data frame of string values|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|f_odered|Frame[String]|sorted dataset by column 1 in decreasing order|

## garch-Function


This is a builtin function that implements GARCH(1,1), a statistical model used in analyzing time-series data where the 
variance

error is believed to be serially autocorrelated
### Usage


```python
garch(X, kmax, momentum, start_stepsize, end_stepsize, start_vicinity, end_vicinity, sim_seed, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|---|The input Matrix to apply Arima on.|
|kmax|Integer|---|Number of iterations|
|momentum|Double|---|Momentum for momentum-gradient descent (set to 0 to deactivate)|
|start_stepsize|Double|---|Initial gradient-descent stepsize|
|end_stepsize|Double|---|gradient-descent stepsize at end (linear descent)|
|start_vicinity|Double|---|proportion of randomness of restart-location for gradient descent at beginning|
|end_vicinity|Double|---|same at end (linear decay)|
|sim_seed|Integer|---|seed for simulation of process on fitted coefficients|
|verbose|Boolean|---|verbosity, comments during fitting|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|fitted_X|Matrix[Double]|simulated garch(1,1) process on fitted coefficients|
|fitted_var_hist|Matrix[Double]|variances of simulated fitted process|
|best_a0|Double|onstant term of fitted process|
|best_arch_coef|Double|1-st arch-coefficient of fitted process|
|best_var_coef|Double|1-st garch-coefficient of fitted process|

## gaussianClassifier-Function


Computes the parameters needed for Gaussian Classification.

Thus it computes the following per class: the prior probability,

the inverse covariance matrix, the mean per feature and the determinant

of the covariance matrix. Furthermore (if not explicitely defined), it

adds some small smoothing value along the variances, to prevent

numerical errors / instabilities.
### Usage


```python
gaussianClassifier(D, C, varSmoothing, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|D|Matrix[Double]|---|Input matrix (training set)|
|C|Matrix[Double]|---|Target vector|
|varSmoothing|Double|1e-9|Smoothing factor for variances|
|verbose|Boolean|TRUE|Print accuracy of the training set|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|classPriors|Matrix[Double]|Vector storing the class prior probabilities|
|classMeans|Matrix[Double]|Matrix storing the means of the classes|
|classInvCovariances|List[Unknown]|List of inverse covariance matrices|
|determinants|Matrix[Double]|Vector storing the determinants of the classes|

## getAccuracy-Function


This builtin function compute the weighted and simple accuracy for given predictions
### Usage


```python
getAccuracy(y, yhat, isWeighted)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|y|Double|---|Ground truth (Actual Labels)|
|yhat|Double|---|predictions (Predicted labels)|
|isWeighted|Boolean|FALSE|flag for weighted or non-weighted accuracy calculation|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|accuracy|Double|accuracy of the predicted labels|

## glm-Function


This script solves GLM regression using NEWTON/FISHER scoring with trust regions. The glm-function is a flexible

generalization of ordinary linear regression that allows for response variables that have error distribution models.
### Usage


```python
glm(X, Y, if, dfam, vpow, 0.0, link, 1, lpow, -2.0, yneg, icpt, 0, 1, 2, reg, tol, disp, moi, mii)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|matrix X of feature vectors|
|Y|Matrix[Double]|---|matrix Y with either 1 or 2 columns:|
|if|dfam|=|2, Y is 1-column Bernoulli or 2-column Binomial (#pos, #neg)|
|dfam|Int|1|Distribution family code: 1 = Power, 2 = Binomial|
|vpow|Double|0.0|Power for Variance defined as (mean)^power (ignored if dfam != 1):|
|0.0|=|Gaussian,|1.0 = Poisson, 2.0 = Gamma, 3.0 = Inverse Gaussian|
|link|Int|0|Link function code: 0 = canonical (depends on distribution),|
|1|=|Power,|2 = Logit, 3 = Probit, 4 = Cloglog, 5 = Cauchit|
|lpow|Double|1.0|Power for Link function defined as (mean)^power (ignored if link != 1):|
|-2.0|=|1/mu^2,|-1.0 = reciprocal, 0.0 = log, 0.5 = sqrt, 1.0 = identity|
|yneg|Double|0.0|Response value for Bernoulli "No" label, usually 0.0 or -1.0|
|icpt|Int|0|Intercept presence, X columns shifting and rescaling:|
|0|=|no|intercept, no shifting, no rescaling;|
|1|=|add|intercept, but neither shift nor rescale X;|
|2|=|add|intercept, shift & rescale X columns to mean = 0, variance = 1|
|reg|Double|0.0|Regularization parameter (lambda) for L2 regularization|
|tol|Double|0.000001|Tolerance (epsilon)|
|disp|Double|0.0|(Over-)dispersion value, or 0.0 to estimate it from data|
|moi|Int|200|Maximum number of outer (Newton / Fisher Scoring) iterations|
|mii|Int|0|Maximum number of inner (Conjugate Gradient) iterations, 0 = no maximum|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|beta|Matrix[Double]|Matrix beta, whose size depends on icpt:|
|icpt=0:|ncol(X)|x 1;  icpt=1: (ncol(X) + 1) x 1;  icpt=2: (ncol(X) + 1) x 2|

## gmm-Function


The gmm-function implements builtin Gaussian Mixture Model with four different types of covariance matrices
### Usage


```python
gmm(X, n_components, model, "EEE":, "VVI":, "VII":, init_param, iterations, reg_covar, tol)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X|
|n_components|Integer|3|Number of n_components in the Gaussian mixture model|
|model|String|"VVV"|"VVV": unequal variance (full),each component has its own general covariance matrix|
|"EEE":|equal|variance|(tied), all components share the same general covariance matrix|
|"VVI":|spherical,|unequal|volume (diag), each component has its own diagonal|
|"VII":|spherical,|equal|volume (spherical), each component has its own single variance|
|init_param|String|"kmeans"|initialize weights with "kmeans" or "random"|
|iterations|Integer|100|Number of iterations|
|reg_covar|Double|1e-6|regularization parameter for covariance matrix|
|tol|Double|0.000001|tolerance value for convergence|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|labels|Matrix[Double]|Prediction matrix|
|predict_prob|Matrix[Double]||
|df|Integer|Number of estimated parameters|
|bic|Double|Bayesian information criterion for best iteration|
|mu|Matrix[Double]|fitted clusters mean|
|weight|Matrix[Double]|A matrix whose [i,k]th entry is the probability that observation i in the test data|
|belongs|to|the kth class|

## gmmPredict-Function


This function is a Prediction function for a Gaussian Mixture Model (gmm).
### Usage


```python
gmmPredict(X, weight, mu, precisions_cholesky, model)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X (instances to be clustered)|
|weight|Matrix[Double]|---|Weight of learned model|
|mu|Matrix[Double]|---|fitted clusters mean|
|precisions_cholesky|Matrix[Double]|---|fitted precision matrix for each mixture|
|model|String|---|fitted model|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|predict|Double|predicted cluster labels|
|posterior_prob|Double|probabilities of belongingness|

## gnmf-Function


References:
### Usage


```python
gnmf(X, rnk, eps, maxi)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|required|Matrix of feature vectors.|
|rnk|Integer|required|Number of components into which matrix X is to be factored|
|eps|Double|1e-8|Tolerance|
|maxi|Integer|10|Maximum number of conjugate gradient iterations|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|W|Matrix[Double]|List of pattern matrices, one for each repetition|
|H|Matrix[Double]|List of amplitude matrices, one for each repetition|

## gridSearch-Function


The gridSearch-function is used to find the optimal hyper-parameters of a model which results in the most

accurate predictions. This function takes train and eval functions by name.
### Usage


```python
gridSearch(X, y, train, predict, numB, size, params, paramValues, columnvectors, trainArgs, gridSearch, not, predictArgs, gridSearch, not, cv, cvk, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|required|Input feature matrix|
|y|Matrix[Double]|required|Input Matrix of vectors.|
|train|String|required|Name ft of the train function to call via ft(trainArgs)|
|predict|String|required|Name fp of the loss function to call via fp((predictArgs,B))|
|numB|Integer|---|Maximum number of parameters in model B (pass the maximum because the|
|size|of|B|may vary with parameters like icpt|
|params|List[String]|required|List of varied hyper-parameter names|
|paramValues|List[Unknown]|List|of matrices providing the parameter values as|
|columnvectors|for|position-aligned|hyper-parameters in 'params'|
|trainArgs|List[Unknown]|named|List of arguments to pass to the 'train' function, where|
|gridSearch|replaces|enumerated|hyper-parameter by name, if|
|not|provided|or|an empty list, the lm parameters are used|
|predictArgs|List[Unknown]|List|of arguments to pass to the 'predict' function, where|
|gridSearch|appends|the|trained models at the end, if|
|not|provided|or|an empty list, list(X, y) is used instead|
|cv|Boolean|FALSE|flag enabling k-fold cross validation, otherwise training loss|
|cvk|Integet|5|if cv=TRUE, specifies the the number of folds, otherwise ignored|
|verbose|Boolean|TRUE|flag for verbose debug output|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|B|Matrix[Double]|Matrix[Double]the trained model with minimal loss (by the 'predict' function)|
|opt|Matrix[Double]|one-row frame w/ optimal hyperparameters (by 'params' position)|

## hospitalResidencyMatch-Function


This script computes a solution for the hospital residency match problem.
### Usage


```python
hospitalResidencyMatch(R, It, H, It, capacity, It, i.e., i.e., with)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|R|Matrix|---|Residents matrix R.|
|It|must|be|an ORDERED  matrix.|
|H|Matrix|---|Hospitals matrix H.|
|It|must|be|an UNORDRED matrix.|
|capacity|Matrix|---|capacity of Hospitals matrix C.|
|It|must|be|a [n*1] matrix with non zero values.|
|i.e.|the|leftmost|value in a row is the most preferred partner's index.|
|i.e.|the|leftmost|value in a row in P is the preference value for the acceptor|
|with|index|1|and vice-versa (higher is better).|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|residencyMatch|Matrix|Result Matrix|
|If|cell|[i,j] is non-zero, it means that Resident i has matched with Hospital j.|
|Further,|if|cell [i,j] is non-zero, it holds the preference value that led to the match.|
|hospitalMatch|Matrix|Result Matrix|
|If|cell|[i,j] is non-zero, it means that Resident i has matched with Hospital j.|
|Further,|if|cell [i,j] is non-zero, it holds the preference value that led to the match.|

## hyperband-Function


The hyperband-function is used for hyper parameter optimization and is based on multi-armed bandits and early

elimination. Through multiple parallel brackets and consecutive trials it will return the hyper parameter combination

which performed best on a validation dataset. A set of hyper parameter combinations is drawn from uniform distributions

with given ranges; Those make up the candidates for hyperband. Notes:

hyperband is hard-coded for lmCG, and uses lmPredict for validation

hyperband is hard-coded to use the number of iterations as a resource

hyperband can only optimize continuous hyperparameters
### Usage


```python
hyperband(X_train, y_train, X_val, y_val, params, paramRanges, One, R, eta, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X_train|Matrix[Double]|required|Input Matrix of training vectors|
|y_train|Matrix[Double]|required|Labels for training vectors|
|X_val|Matrix[Double]|required|Input Matrix of validation vectors|
|y_val|Matrix[Double]|required|Labels for validation vectors|
|params|List[String]|required|List of parameters to optimize|
|paramRanges|Matrix[Double]|required|The min and max values for the uniform distributions to draw from.|
|One|row|per|hyper parameter, first column specifies min, second column max value.|
|R|Scalar[int]|81|Controls number of candidates evaluated|
|eta|Scalar[int]|3|Determines fraction of candidates to keep after each trial|
|verbose|Boolean|TRUE|If TRUE print messages are activated|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|bestWeights|Matrix[Double]|1-column matrix of weights of best performing candidate|
|bestHyperParams|Frame[Unknown]|hyper parameters of best performing candidate|

## img_brightness-Function


The img_brightness-function is an image data augumentation function. It changes the brightness of the image.
### Usage


```python
img_brightness(img_in, value, channel_max)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input matrix/image|
|value|Double|---|The amount of brightness to be changed for the image|
|channel_max|Integer|---|Maximum value of the brightness of the image|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output matrix/image|

## img_crop-Function


The img_crop-function is an image data augumentation function. It cuts out a subregion of an image.
### Usage


```python
img_crop(img_in, w, h, x_offset, y_offset)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input matrix/image|
|w|Integer|---|The width of the subregion required|
|h|Integer|---|The height of the subregion required|
|x_offset|Integer|---|The horizontal coordinate in the image to begin the crop operation|
|y_offset|Integer|---|The vertical coordinate in the image to begin the crop operation|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Cropped matrix/image|

## img_cutout-Function


Image Cutout function replaces a rectangular section of an image with a constant value.
### Usage


```python
img_cutout(img_in, x, y, width, height, fill_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image as 2D matrix with top left corner at [1, 1]|
|x|Int|---|Column index of the top left corner of the rectangle (starting at 1)|
|y|Int|---|Row index of the top left corner of the rectangle (starting at 1)|
|width|Int|---|Width of the rectangle (must be positive)|
|height|Int|---|Height of the rectangle (must be positive)|
|fill_value|Double|---|The value to set for the rectangle|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image as 2D matrix with top left corner at [1, 1]|

## img_invert-Function


This is an image data augumentation function. It inverts an image.
### Usage


```python
img_invert(img_in, max_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image|
|max_value|Double|---|The maximum value pixels can have|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image|

## img_mirror-Function


This function is an image data augumentation function. It flips an image on the X (horizontal) or Y (vertical) axis.
### Usage


```python
img_mirror(img_in, max_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input matrix/image|
|max_value|Double|---|The maximum value pixels can have|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Flipped matrix/image|

## img_posterize-Function


The Image Posterize function limits pixel values to 2^bits different values in the range [0, 255].

Assumes the input image can attain values in the range [0, 255].
### Usage


```python
img_posterize(img_in, bits, 1)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image|
|bits|Int|---|The number of bits keep for the values.|
|1|means|black|and white, 8 means every integer between 0 and 255.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image|

## img_rotate-Function


The Image Rotate function rotates the input image counter-clockwise around the center.

Uses nearest neighbor sampling.
### Usage


```python
img_rotate(img_in, radians, fill_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image as 2D matrix with top left corner at [1, 1]|
|radians|Double|---|The value by which to rotate in radian.|
|fill_value|Double|---|The background color revealed by the rotation|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image as 2D matrix with top left corner at [1, 1]|

## img_sample_pairing-Function


the Image Sample Pairing function blends two images together.
### Usage


```python
img_sample_pairing(img_in1, img_in2, weight, 0)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in1|Matrix[Double]|---|First input image|
|img_in2|Matrix[Double]|---|Second input image|
|weight|Double|---|The weight given to the second image.|
|0|means|only|img_in1, 1 means only img_in2 will be visible|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image|

## img_shear-Function


This function applies a shearing transformation to an image.

Uses nearest neighbor sampling.
### Usage


```python
img_shear(img_in, shear_x, shear_y, fill_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image as 2D matrix with top left corner at [1, 1]|
|shear_x|Double|---|Shearing factor for horizontal shearing|
|shear_y|Double|---|Shearing factor for vertical shearing|
|fill_value|Double|---|The background color revealed by the shearing|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image as 2D matrix with top left corner at [1, 1]|

## img_transform-Function


the Image Transform function applies an affine transformation to an image.

Optionally resizes the image (without scaling).

Uses nearest neighbor sampling.
### Usage


```python
img_transform(img_in, out_w, out_h, a,b,c,d,e,f, fill_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image as 2D matrix with top left corner at [1, 1]|
|out_w|Integer|---|Width of the output image|
|out_h|Integer|---|Height of the output image|
|a,b,c,d,e,f|Double|---|The first two rows of the affine matrix in row-major order|
|fill_value|Double|---|The background of the image|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix|Output image as 2D matrix with top left corner at [1, 1]|

## img_translate-Function


the Image Translate function translates the image.

Optionally resizes the image (without scaling).

Uses nearest neighbor sampling.
### Usage


```python
img_translate(img_in, offset_x, offset_y, out_w, out_h, fill_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|img_in|Matrix[Double]|---|Input image as 2D matrix with top left corner at [1, 1]|
|offset_x|Double|---|The distance to move the image in x direction|
|offset_y|Double|---|The distance to move the image in y direction|
|out_w|Int|---|Width of the output image|
|out_h|Int|---|Height of the output image|
|fill_value|Double|---|The background of the image|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|img_out|Matrix[Double]|Output image as 2D matrix with top left corner at [1, 1]|

## imputeByFD-Function


Implements builtin for imputing missing values from observed values (if exist) using robust functional dependencies
### Usage


```python
imputeByFD(X, source, target, threshold)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X|
|source|Integer|---|source attribute to use for imputation and error correction|
|target|Integer|---|attribute to be fixed|
|threshold|Double|---|threshold value in interval [0, 1] for robust FDs|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|Matrix with possible imputations|

## imputeByMean-Function


impute the data by mean value and if the feature is categorical then by mode value

Related to [SYSTEMDS-2662] dependency function for cleaning pipelines
### Usage


```python
imputeByMean(X, mask)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix (Recoded Matrix for categorical features)|
|mask|Matrix[Double]|---|A 0/1 row vector for identifying numeric (0) and categorical features (1)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|imputed dataset|

## imputeByMedian-Function


impute the data by median value and if the feature is categorical then by mode value

Related to [SYSTEMDS-2662] dependency function for cleaning pipelines
### Usage


```python
imputeByMedian(X, mask)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix (Recoded Matrix for categorical features)|
|mask|Matrix[Double]|---|A 0/1 row vector for identifying numeric (0) and categorical features (1)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|imputed dataset|

## imputeByMode-Function


This function impute the data by mode value

Related to [SYSTEMDS-2902] dependency function for cleaning pipelines
### Usage


```python
imputeByMode(X)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix (Recoded Matrix for categorical features)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|imputed dataset|

## intersect-Function


The intersect function implements set intersection for numeric data.
### Usage


```python
intersect(X, Y)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|matrix X, set A|
|Y|Matrix[Double]|---|matrix Y, set B|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|R|Matrix[Double]|intersection matrix, set of intersecting items|

## km-Function


Builtin function that implements the analysis of survival data with KAPLAN-MEIER estimates
### Usage


```python
km(X, timestamps,, number, TE, information, GI, SI, alpha, function, err_type, conf_type, upper, corresponds, complementary, test_type, perform, "log-rank")
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input matrix X containing the survival data:|
|timestamps,|whether|event|occurred (1) or data is censored (0), and a|
|number|of|factors|(categorical features) for grouping and/or stratifying|
|TE|Matrix[Double]|---|Column indices of X which contain timestamps (first entry) and event|
|information|(second|entry)||
|GI|Matrix[Double]|---|Column indices of X corresponding to the factors to be used for grouping|
|SI|Matrix[Double]|---|Column indices of X corresponding to the factors to be used for stratifying|
|alpha|Double|0.05|Parameter to compute 100*(1-alpha)% confidence intervals for the survivor|
|function|and|its|median|
|err_type|String|"greenwood"|Parameter to specify the error type according to "greenwood" (the default) or "peto"|
|conf_type|String|"log"|Parameter to modify the confidence interval; "plain" keeps the lower and|
|upper|bound|of|the confidence interval unmodified, "log" (the default)|
|corresponds|to|logistic|transformation and "log-log" corresponds to the|
|complementary|log-log|transformation||
|test_type|String|"none"|If survival data for multiple groups is available specifies which test to|
|perform|for|comparing|survival data across multiple groups: "none" (the default)|
|"log-rank"|or|"wilcoxon"|test|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|O|Matrix[Double]|Matrix KM whose dimension depends on the number of groups (denoted by g) and|
|strata|(denoted|by s) in the data:|
|each|collection|of 7 consecutive columns in KM corresponds to a unique|
|combination|of|groups and strata in the data with the following schema|
|1.|col:|timestamp|
|2.|col:|no. at risk|
|3.|col:|no. of events|
|4.|col:|Kaplan-Meier estimate of survivor function surv|
|5.|col:|standard error of surv|
|6.|col:|lower 100*(1-alpha)% confidence interval for surv|
|7.|col:|upper 100*(1-alpha)% confidence interval for surv|
|M|Matrix[Double]|Matrix M whose dimension depends on the number of groups (g) and strata (s) in|
|the|data|(k denotes the number of factors used for grouping  ,i.e., ncol(GI) and|
|l|denotes|the number of factors used for stratifying, i.e., ncol(SI))|
|M[,1:k]:|unique|combination of values in the k factors used for grouping|
|M[,(k+1):(k+l)]:|unique|combination of values in the l factors used for stratifying|
|M[,k+l+1]:|total|number of records|
|M[,k+l+2]:|total|number of events|
|M[,k+l+3]:|median|of surv|
|M[,k+l+4]:|lower|100*(1-alpha)% confidence interval of the median of surv|
|M[,k+l+5]:|upper|100*(1-alpha)% confidence interval of the median of surv|
|If|the|number of groups and strata is equal to 1, M will have 4 columns with|
|M[,1]:|total|number of events|
|M[,2]:|median|of surv|
|M[,3]:|lower|100*(1-alpha)% confidence interval of the median of surv|
|M[,4]:|upper|100*(1-alpha)% confidence interval of the median of surv|
|T_GROUPS_OE|Matrix[Double]|If survival data from multiple groups available and ttype=log-rank or wilcoxon,|
|a|1|x 4 matrix T and an g x 5 matrix T_GROUPS_OE with|
|T_GROUPS_OE[,1]|=|no. of events|
|T_GROUPS_OE[,2]|=|observed value (O)|
|T_GROUPS_OE[,3]|=|expected value (E)|
|T_GROUPS_OE[,4]|=|(O-E)^2/E|
|T_GROUPS_OE[,5]|=|(O-E)^2/V|
|T[1,1]|=|no. of groups|
|T[1,2]|=|degree of freedom for Chi-squared distributed test statistic|
|T[1,3]|=|test statistic|
|T[1,4]|=|P-value|

## kmeans-Function


Builtin function that implements the k-Means clustering algorithm
### Usage


```python
kmeans(X, k, runs, max_iter, eps, is_verbose, avg_sample_size_per_centroid, seed, random)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|The input Matrix to do KMeans on.|
|k|Int|10|Number of centroids|
|runs|Int|10|Number of runs (with different initial centroids)|
|max_iter|Int|1000|Maximum number of iterations per run|
|eps|Double|0.000001|Tolerance (epsilon) for WCSS change ratio|
|is_verbose|Boolean|FALSE|do not print per-iteration stats|
|avg_sample_size_per_centroid|Int|50|Average number of records per centroid in data samples|
|seed|Int|-1|The seed used for initial sampling. If set to -1|
|random|seeds|are|selected.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|String|"Y.mtx"  The mapping of records to centroids|
|C|String|"C.mtx"  The output matrix with the centroids|

## kmeansPredict-Function


Builtin function that does predictions based on a set of centroids provided.
### Usage


```python
kmeansPredict(X, C)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|The input Matrix to do KMeans on.|
|C|Matrix[Double]|---|The input Centroids to map X onto.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|"Y.mtx"  The mapping of records to centroids|

## knnbf-Function


This script implements KNN (K Nearest Neighbor) algorithm.
### Usage


```python
knnbf(X, T, k_value)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|---|
|T|Matrix[Double]|---|---|
|k_value|Integer|5|---|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|NNR|Matrix[Double]|---|

## l2svm-Function


Builtin function Implements binary-class SVM with squared slack variables
### Usage


```python
l2svm(X, Y, intercept, epsilon, than, lambda, maxIterations, maxii, verbose, columnId, usefull)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix|---|matrix X of feature vectors|
|Y|Matrix|---|matrix Y of class labels have to be a single column|
|intercept|Boolean|False|No Intercept ( If set to TRUE then a constant bias column is added to X)|
|epsilon|Double|0.001|Procedure terminates early if the reduction in objective function value is less|
|than|epsilon|(tolerance)|times the initial objective function value.|
|lambda|Double|1.0|Regularization parameter (lambda) for L2 regularization|
|maxIterations|Int|100|Maximum number of conjugate gradient iterations|
|maxii|Int|20|-|
|verbose|Boolean|FALSE|Set to true if one wants print statements updating on loss.|
|columnId|Int|-1|The column Id used if one wants to add a ID to the print statement, Specificly|
|usefull|when|L2SVM|is used in MSVM.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|model|Matrix[Double]|model matrix|

## l2svmPredict-Function


Builtin function Implements binary-class SVM with squared slack variables.
### Usage


```python
l2svmPredict(X, W, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|---|matrix X of feature vectors to classify|
|W|Double|---|matrix of the trained variables|
|verbose|Boolean|FALSE|Set to true if one wants print statements.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|YRaw|Matrix[Double]|Classification Labels Raw, meaning not modified to clean|
|Labeles|of|1's and -1's|
|Y|Matrix[Double]|Classification Labels Maxed to ones and zeros.|

## lasso-Function


Builtin function for the SpaRSA algorithm to perform lasso regression
### Usage


```python
lasso(X, y, tol, M, tau, maxi)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|input feature matrix|
|y|Matrix[Double]|---|matrix Y columns of the design matrix|
|tol|Double|1e-15|target convergence tolerance|
|M|Integer|5|history length|
|tau|Double|1|regularization component|
|maxi|Integer|100|maximum number of iterations until convergence|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|w|Matrix[Double]|model matrix|

## lenetPredict-Function


This builtin function makes prediction given data and trained LeNet model
### Usage


```python
lenetPredict(model, X, C, Hin, Win, batch_size)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|model|List[unknown]|---|Trained LeNet model|
|X|Matrix[Double]|---|Input data matrix, of shape (N, C*Hin*Win)|
|C|Integer|---|Number of input channels|
|Hin|Integer|---|Input height|
|Win|Integer|---|Input width|
|batch_size|Integer|---|Batch size|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|probs|Matrix[Double]|Predicted values|

## lenetTrain-Function


This builtin function trains LeNet CNN. The architecture of the

networks is:conv1 -> relu1 -> pool1 -> conv2 -> relu2 -> pool2 ->

affine3 -> relu3 -> affine4 -> softmax
### Usage


```python
lenetTrain(X, Y, X_val, Y_val, C, Hin, Win, batch_size, epochs, lr, mu, decay, lambda, seed, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[double]|---|Input data matrix, of shape (N, C*Hin*Win)|
|Y|Matrix[Double]|---|Target matrix, of shape (N, K)|
|X_val|Matrix[double]|---|Validation data matrix, of shape (N, C*Hin*Win)|
|Y_val|Matrix[Double]|---|Validation target matrix, of shape (N, K)|
|C|Integer|---|Number of input channels (dimensionality of input depth)|
|Hin|Integer|---|Input width|
|Win|Integer|---|Input height|
|batch_size|Integer|64|Batch size|
|epochs|Integer|20|Number of epochs|
|lr|Double|0.01|Learning rate|
|mu|Double|0.9|Momentum value|
|decay|Double|0.95|Learning rate decay|
|lambda|Double|5e-04|Regularization strength|
|seed|Integer|-1|Seed for model initialization|
|verbose|Boolean|FALSE|Flag indicates if function should print to stdout|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|model|List[unknown]|Trained model which can be used in lenetPredict|

## lm-Function


The lm-function solves linear regression using either the direct solve method or the conjugate gradient

algorithm depending on the input size of the matrices (See lmDS-function and lmCG-function respectively).
### Usage


```python
lm(X, y, icpt, reg, for, tol, norm, maxi, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors.|
|y|Matrix[Double]|---|1-column matrix of response values.|
|icpt|Integer|0|Intercept presence, shifting and rescaling the columns of X|
|reg|Double|1e-7|Regularization constant (lambda) for L2-regularization. set to nonzero|
|for|highly|dependant/sparse/numerous|features|
|tol|Double|1e-7|Tolerance (epsilon); conjugate gradient procedure terminates early if L2|
|norm|of|the|beta-residual is less than tolerance * its initial norm|
|maxi|Integer|0|Maximum number of conjugate gradient iterations. 0 = no maximum|
|verbose|Boolean|TRUE|If TRUE print messages are activated|

### Returns

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|B|String|"B.mtx"|The model fit|

## lmCG-Function


The lmCG function solves linear regression using the conjugate gradient algorithm
### Usage


```python
lmCG(X, y, icpt, reg, for, tol, norm, maxi, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors.|
|y|Matrix[Double]|---|1-column matrix of response values.|
|icpt|Integer|0|Intercept presence, shifting and rescaling the columns of X|
|reg|Double|1e-7|Regularization constant (lambda) for L2-regularization. set to nonzero|
|for|highly|dependant/sparse/numerous|features|
|tol|Double|1e-7|Tolerance (epsilon); conjugate gradient procedure terminates early if L2|
|norm|of|the|beta-residual is less than tolerance * its initial norm|
|maxi|Integer|0|Maximum number of conjugate gradient iterations. 0 = no maximum|
|verbose|Boolean|TRUE|If TRUE print messages are activated|

### Returns

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|B|String|"B.mtx"|The model fit|

## lmDS-Function


The lmDC function solves linear regression using  the direct solve method
### Usage


```python
lmDS(X, y, icpt, reg, for, tol, norm, maxi, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors.|
|y|Matrix[Double]|---|1-column matrix of response values.|
|icpt|Integer|0|Intercept presence, shifting and rescaling the columns of X|
|reg|Double|1e-7|Regularization constant (lambda) for L2-regularization. set to nonzero|
|for|highly|dependant/sparse/numerous|features|
|tol|Double|1e-7|Tolerance (epsilon); conjugate gradient procedure terminates early if L2|
|norm|of|the|beta-residual is less than tolerance * its initial norm|
|maxi|Integer|0|Maximum number of conjugate gradient iterations. 0 = no maximum|
|verbose|Boolean|TRUE|If TRUE print messages are activated|

### Returns

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|B|String|"B.mtx"|The model fit|

## lmPredict-Function


The lmPredict-function predicts the class of a feature vector
### Usage


```python
lmPredict(X, B, ytest, if, icpt, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors|
|B|Matrix[Double]|---|1-column matrix of weights.|
|ytest|Matrix[Double]|---|test labels, used only for verbose output. can be set to matrix(0,1,1)|
|if|verbose|output|is not wanted|
|icpt|Integer|0|Intercept presence, shifting and rescaling the columns of X|
|verbose|Boolean|TRUE|If TRUE print messages are activated|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|yhat|String|1-column matrix of classes|

## logSumExp-Function


Built-in LOGSUMEXP
### Usage


```python
logSumExp(X, margin, if, if)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|matrix M|
|margin|String|none|if the logsumexp of rows is required set margin = "row"|
|if|the|logsumexp|of columns is required set margin = "col"|
|if|set|to|"none" then a single scalar is returned computing logsumexp of matrix|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|output|Matrix[Double]|a 1*1 matrix, row vector or column vector depends on margin value|

## matrixProfile-Function


References:

Yan Zhu et al.. 2018.

Matrix Profile XI: SCRIMP++: Time Series Motif Discovery at Interactive Speeds.
### Usage


```python
matrixProfile(ts, window_size, sample_percent, between, computes, is_verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|ts|Matrix[Double]|---|Time series to profile|
|window_size|Integer|4|Sliding window size|
|sample_percent|Double|1.0|Degree of approximation|
|between|zero|and|one (1|
|computes|the|exact|solution)|
|is_verbose|Boolean|False|Print debug information|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|profile|Matrix[Double]|The computed matrix profile|
|profile_index|Matrix[Double]|Indices of least distances|

## mdedup-Function


Implements builtin for deduplication using matching dependencies (e.g. Street 0.95, City 0.90 -> ZIP 1.0)

and Jaccard distance.
### Usage


```python
mdedup(X, LHSfeatures, (e.g., LHSthreshold, RHSfeatures, RHSthreshold, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Frame|--|Input Frame X|
|LHSfeatures|Matrix[Integer]|--|A matrix 1xd with numbers of columns for MDs|
|(e.g.|Street|0.95,|City 0.90 -> ZIP 1.0)|
|LHSthreshold|Matrix[Double]|--|A matrix 1xd with threshold values in interval [0, 1] for MDs|
|RHSfeatures|Matrix[Integer]|--|A matrix 1xd with numbers of columns for MDs|
|RHSthreshold|Matrix[Double]|--|A matrix 1xd with threshold values in interval [0, 1] for MDs|
|verbose|Boolean|--|To print the output|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|MD|Matrix[Double]|Matrix nx1 of duplicates|

## mice-Function


This Builtin function implements multiple imputation using Chained Equations (MICE)
### Usage


```python
mice(X, cMask, iter, threshold, if, only, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix (Recoded Matrix for categorical features)|
|cMask|Matrix[Double]|---|A 0/1 row vector for identifying numeric (0) and categorical features (1)|
|iter|Integer|3|Number of iteration for multiple imputations|
|threshold|Double|0.8|confidence value [0, 1] for robust imputation, values will only be imputed|
|if|the|predicted|value has probability greater than threshold,|
|only|applicable|for|categorical data|
|verbose|Boolean|FALSE|Boolean value.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|output|Matrix[Double]|imputed dataset|

## msvm-Function


Implements builtin multiclass SVM with squared slack variables,

learns one-against-the-rest binary-class classifiers by making a function call to l2SVM
### Usage


```python
msvm(X, Y, intercept, num_classes, epsilon, value, lambda, maxIterations, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|matrix X of feature vectors|
|Y|Matrix[Double]|---|matrix Y of class labels|
|intercept|Boolean|False|No Intercept ( If set to TRUE then a constant bias column is added to X)|
|num_classes|integer|10|Number of classes|
|epsilon|Double|0.001|Procedure terminates early if the reduction in objective function|
|value|is|less|than epsilon (tolerance) times the initial objective function value.|
|lambda|Double|1.0|Regularization parameter (lambda) for L2 regularization|
|maxIterations|Int|100|Maximum number of conjugate gradient iterations|
|verbose|Boolean|False|Set to true to print while training.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|model|Matrix[Double]|model matrix|

## msvmPredict-Function


This Scripts helps in applying an trained MSVM
### Usage


```python
msvmPredict(X, W)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|matrix X of feature vectors to classify|
|W|Matrix[Double]|---|matrix of the trained variables|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|YRaw|Matrix[Double]|Classification Labels Raw, meaning not modified to clean|
|Labeles|of|1's and -1's|
|Y|Matrix[Double]|Classification Labels Maxed to ones and zeros.|

## multiLogReg-Function


Solves Multinomial Logistic Regression using Trust Region method.
### Usage


```python
multiLogReg(X, Y, icpt, no, rescale, tol, reg, maxi, maxii, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Location to read the matrix of feature vectors|
|Y|Matrix[Double]|---|Location to read the matrix with category labels|
|icpt|Integer|0|Intercept presence, shifting and rescaling X columns: 0 = no intercept,|
|no|shifting,|no|rescaling; 1 = add intercept, but neither shift nor|
|rescale|X;|2|= add intercept, shift & rescale X columns to mean = 0, variance = 1|
|tol|Double|0.000001|tolerance ("epsilon")|
|reg|Double|0.0|regularization parameter (lambda = 1/C); intercept is not regularized|
|maxi|Integer|100|max. number of outer (Newton) iterations|
|maxii|Integer|0|max. number of inner (conjugate gradient) iterations, 0 = no max|
|verbose|Boolean|FALSE|flag specifying if logging information should be printed|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|betas|Matrix[Double]|regression betas as output for prediction|

## multiLogRegPredict-Function


THIS SCRIPT APPLIES THE ESTIMATED PARAMETERS OF MULTINOMIAL LOGISTIC REGRESSION TO A NEW (TEST) DATASET

Matrix M of predicted means/probabilities, some statistics in CSV format (see below)
### Usage


```python
multiLogRegPredict(X, B, Y, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix X|
|B|Matrix[Double]|---|Regression parameters betas|
|Y|Matrix[Double]|---|Response vector Y|
|verbose|Boolean|FALSE|flag specifying if logging information should be printed|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Matrix[Double]|Matrix M of predicted means/probabilities|
|predicted_Y|Matrix[Double]|Predicted response vector|
|accuracy|Double|scalar value of accuracy|

## naiveBayes-Function


The naiveBayes-function computes the class conditional probabilities and class priors.
### Usage


```python
naiveBayes(D, C, Laplace, Verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|D|Matrix[Double]|required|One dimensional column matrix with N rows.|
|C|Matrix[Double]|required|One dimensional column matrix with N rows.|
|Laplace|Double|1|Any Double value.|
|Verbose|Boolean|TRUE|Boolean value.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|prior|Matrix[Double]|Class priors, One dimensional column matrix with N rows.|
|classConditionals|Matrix[Double]|Class conditional probabilites, One dimensional column matrix with N rows.|

## naiveBayesPredict-Function


The naiveBaysePredict-function predicts the scoring with a naive Bayes model.
### Usage


```python
naiveBayesPredict(X, P, C)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of test data with N rows.|
|P|Matrix[Double]|---|Class priors, One dimensional column matrix with N rows.|
|C|Matrix[Double]|---|Class conditional probabilities, matrix with N rows|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|A matrix containing the top-K item-ids with highest predicted ratings.|
|YRaw|Matrix[Double]|A matrix containing predicted ratings.|

## na_locf-Function


Builtin function for imputing missing values using forward fill and backward fill techniques
### Usage


```python
na_locf(X, option, String, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X|
|option|String|"locf"|String "locf" (last observation moved forward) to do forward fill|
|String|"nocb"|(next|observation carried backward) to do backward fill|
|verbose|Boolean|FALSE|to print output on screen|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|output|Matrix[Double]|Matrix with no missing values|

## normalize-Function


The normalize-function normalises the values of a matrix by changing the dataset to use a common scale.

This is done while preserving differences in the ranges of values. The output is a matrix of values in range [0,1].
### Usage


```python
normalize(X)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|1-column matrix of normalized values.|

## outlier-Function


This outlier-function takes a matrix data set as input from where it determines

which point(s) have the largest difference from mean.
### Usage


```python
outlier(X, opposite, (0)FALSE)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of Recoded dataset for outlier evaluation|
|opposite|Boolean|---|(1)TRUE for evaluating outlier from upper quartile range,|
|(0)FALSE|for|evaluating|outlier from lower quartile range|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|matrix indicating outlier values|

## outlierByArima-Function


Built-in function for detecting and repairing outliers in time series, by training an ARIMA model

and classifying values that are more than k standard-deviations away from the predicated values as outliers.
### Usage


```python
outlierByArima(X, k, repairMethod, 2, p, d, q, P, D, Q, s, include_mean, solver)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Double|---|Matrix X|
|k|Double|3|threshold values 1, 2, 3 for 68%, 95%, 99.7% respectively (3-sigma rule)|
|repairMethod|Integer|1|values: 0 = delete rows having outliers, 1 = replace outliers as zeros|
|2|=|replace|outliers as missing values|
|p|Int|0|non-seasonal AR order|
|d|Int|0|non-seasonal differencing order|
|q|Int|0|non-seasonal MA order|
|P|Int|0|seasonal AR order|
|D|Int|0|seasonal differencing order|
|Q|Int|0|seasonal MA order|
|s|Int|1|period in terms of number of time-steps|
|include_mean|Bool|FALSE||
|solver|String|"jacobi"|solver, is either "cg" or "jacobi"|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X_corrected|Matrix[Double]|Matrix X with no outliers|

## outlierByIQR-Function


Builtin function for detecting and repairing outliers using standard deviation
### Usage


```python
outlierByIQR(X, k, isIterative, repairMethod, 1, 2, max_iterations, n, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X|
|k|Double|1.5|a constant used to discern outliers k*IQR|
|isIterative|Boolean|TRUE|iterative repair or single repair|
|repairMethod|Integer|1|values: 0 = delete rows having outliers,|
|1|=|replace|outliers with zeros|
|2|=|replace|outliers as missing values|
|max_iterations|Integer|0|values: 0 = arbitrary number of iteraition until all outliers are removed,|
|n|=|any|constant defined by user|
|verbose|Boolean|FALSE|flag specifying if logging information should be printed|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Matrix X with no outliers|

## outlierBySd-Function


Builtin function for detecting and repairing outliers using standard deviation
### Usage


```python
outlierBySd(X, k, repairMethod, 2, max_iterations, n)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X|
|k|Double|3|threshold values 1, 2, 3 for 68%, 95%, 99.7% respectively (3-sigma rule)|
|repairMethod|Integer|1|values: 0 = delete rows having outliers, 1 = replace outliers as  zeros|
|2|=|replace|outliers as missing values|
|max_iterations|Integer|0|values: 0 = arbitrary number of iteration until all outliers are removed,|
|n|=|any|constant defined by user|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Matrix X with no outliers|

## pca-Function


The function Principal Component Analysis (PCA) is used for dimensionality reduction
### Usage


```python
pca(X, K, Center, Scale)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|K|Int|2|Number of reduced dimensions (i.e., columns)|
|Center|Boolean|TRUE|Indicates whether or not to center the feature matrix|
|Scale|Boolean|TRUE|Indicates whether or not to scale the feature matrix|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Xout|Matrix[Double]|Output feature matrix with K columns|
|Mout|Matrix[Double]|Output dominant eigen vectors (can be used for projections)|
|Centering|Matrix[Double]|The column means of the input, subtracted to construct the PCA|
|ScaleFactor|Matrix[Double]|The Scaling of the values, to make each dimension same size.|

## pcaInverse-Function


Principal Component Analysis (PCA) for reconstruction of approximation of the original data.

This methods allows to reconstruct an approximation of the original matrix, and is usefull for

calculating how much information is lost in the PCA.
### Usage


```python
pcaInverse(Y, Clusters, Centering, ScaleFactor)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|Y|Matrix[Double]|---|Input features that have PCA applied to them|
|Clusters|Matrix[Double]|---||
|Centering|Matrix[Double]|empty|matrix  The column means of the PCA model, subtracted to construct the PCA|
|ScaleFactor|Matrix[Double]|empty|matrix  The scaling of each dimension in the PCA model|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X|Matrix[Double]|Output feature matrix reconstructing and approximation of the original matrix|

## pcaTransform-Function


Principal Component Analysis (PCA) for dimensionality reduction prediciton

This method is used to transpose data, which the PCA model was not trained on. To validate how good

The PCA is, and to apply in production.
### Usage


```python
pcaTransform(X, Clusters, Centering, ScaleFactor)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|Clusters|Matrix[Double]|---||
|Centering|Matrix[Double]|empty|matrix   The column means of the PCA model, subtracted to construct the PCA|
|ScaleFactor|Matrix[Double]|empty|matrix   The scaling of each dimension in the PCA model|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Output feature matrix dimensionally reduced by PCA|

## pnmf-Function


The pnmf-function implements Poisson Non-negative Matrix Factorization (PNMF). Matrix X is factorized into two

non-negative matrices, W and H based on Poisson probabilistic assumption. This non-negativity makes the resulting

matrices easier to inspect.
### Usage


```python
pnmf(X, rnk, eps, maxi, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors.|
|rnk|Integer|---|Number of components into which matrix X is to be factored.|
|eps|Double|10^-8|Tolerance|
|maxi|Integer|10|Maximum number of conjugate gradient iterations.|
|verbose|Boolean|TRUE|If TRUE, 'iter' and 'obj' are printed.|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|W|Matrix[Double]|List of pattern matrices, one for each repetition.|
|H|Matrix[Double]|List of amplitude matrices, one for each repetition.|

## ppca-Function


This script performs Probabilistic Principal Component Analysis (PCA) on the given input data.

It is based on paper: sPCA: Scalable Principal Component Analysis for Big Data on Distributed

Platforms. Tarek Elgamal et.al.
### Usage


```python
ppca(X, k, maxi, tolobj, tolrecerr, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix|---|n x m input feature matrix|
|k|Integer|---|indicates dimension of the new vector space constructed from eigen vectors|
|maxi|Integer|---|maximum number of iterations until convergence|
|tolobj|Double|0.00001|objective function tolerance value to stop ppca algorithm|
|tolrecerr|Double|0.02|reconstruction error tolerance value to stop the algorithm|
|verbose|Boolen|TRUE|verbose debug output|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Xout|Matrix|Output feature matrix with K columns|
|Mout|Matrix|Output dominant eigen vectors (can be used for projections)|

## randomForest-Function


This script implement classification random forest with both scale and categorical features.
### Usage


```python
randomForest(X, Y, R, -, -, -, If, bins, depth, num_leaf, num_samples, num_trees, subsamp_rate, Poisson, feature_subset, as, by, impurity)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix|---|Feature matrix X; note that X needs to be both recoded and dummy coded|
|Y|Matrix|---|Label matrix Y; note that Y needs to be both recoded and dummy coded|
|R|Matrix|"|"          Matrix which for each feature in X contains the following information|
|-|R[,1]:|column|ids       TODO pass recorded and binned|
|-|R[,2]:|start|indices|
|-|R[,3]:|end|indices|
|If|R|is|not provided by default all variables are assumed to be scale|
|bins|Int|20|Number of equiheight bins per scale feature to choose thresholds|
|depth|Int|25|Maximum depth of the learned tree|
|num_leaf|Int|10|Number of samples when splitting stops and a leaf node is added|
|num_samples|Int|3000|Number of samples at which point we switch to in-memory subtree building|
|num_trees|Int|10|Number of trees to be learned in the random forest model|
|subsamp_rate|Double|1.0|Parameter controlling the size of each tree in the forest; samples are selected from a|
|Poisson|distribution|with|parameter subsamp_rate (the default value is 1.0)|
|feature_subset|Double|0.5|Parameter that controls the number of feature used as candidates for splitting at each tree node|
|as|a|power|of number of features in the dataset;|
|by|default|square|root of features (i.e., feature_subset = 0.5) are used at each tree node|
|impurity|String|"Gini"|Impurity measure: entropy or Gini (the default)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Matrix|Matrix M containing the learned tree, where each column corresponds to a node|
|in|the|learned tree and each row contains the following information:|
|M[1,j]:|id|of node j (in a complete binary tree)|
|M[2,j]:|tree|id to which node j belongs|
|M[3,j]:|Offset|(no. of columns) to left child of j|
|M[4,j]:|Feature|index of the feature that node j looks at if j is an internal node, otherwise 0|
|M[5,j]:|Type|of the feature that node j looks at if j is an internal node: 1 for scale and 2|
|for|categorical|features,|
|otherwise|the|label that leaf node j is supposed to predict|
|M[6,j]:|1|if j is an internal node and the feature chosen for j is scale, otherwise the|
|size|of|the subset of values|
|stored|in|rows 7,8,... if j is categorical|
|M[7:,j]:|Only|applicable for internal nodes. Threshold the example's feature value is|
|compared|to|is stored at M[7,j] if the feature chosen for j is scale;|
|If|the|feature chosen for j is categorical rows 7,8,... depict the value subset chosen for j|
|C|Matrix|Matrix C containing the number of times samples are chosen in each tree of the random forest|
|S_map|Matrix|Mappings from scale feature ids to global feature ids|
|C_map|Matrix|Mappings from categorical feature ids to global feature ids|

## scale-Function


This function scales and center individual features in the input matrix (column wise.) using z-score to scale the 
values.
### Usage


```python
scale(X, Center, Scale)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|Center|Boolean|TRUE|Indicates whether or not to center the feature matrix|
|Scale|Boolean|TRUE|Indicates whether or not to scale the feature matrix|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Output feature matrix with K columns|
|ColMean|Matrix[Double]|The column means of the input, subtracted if Center was TRUE|
|ScaleFactor|Matrix[Double]|The Scaling of the values, to make each dimension have similar value ranges|

## scaleApply-Function


This function scales and center individual features in the input matrix (column wise.) using the input matrices.
### Usage


```python
scaleApply(X, Centering, ScaleFactor)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|Centering|Matrix[Double]|---|The column means to subtract from X (not done if empty)|
|ScaleFactor|Matrix[Double]|---|The column scaling to multiply with X (not done if empty)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Output feature matrix with K columns|

## selectByVarThresh-Function


This function drops feature with <= thresh variance (by default drop constants).
### Usage


```python
selectByVarThresh(X, thresh)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|matrix[double]|---|Matrix of feature vectors.|
|thresh|Double|0||

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Xp|matrix[double]|Matrix of feature vectors with <= thresh variance.|

## sherlock-Function


This function implements training phase of Sherlock: A Deep Learning Approach to Semantic Data Type Detection
### Usage


```python
sherlock(X_train, y_train)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X_train|matrix[double]|---|maxtrix of feature vectors|
|y_train|matrix[double]|---|matrix Y of class labels of semantic data type|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|cW|matrix[double]|weights (parameters) matrices for character distribtions|
|cb|matrix[double]|biases vectors for character distribtions|
|wW|matrix[double]|weights (parameters) matrices for word embeddings|
|wb|matrix[double]|biases vectors for word embeddings|
|pW|matrix[double]|weights (parameters) matrices for paragraph vectors|
|pb|matrix[double]|biases vectors for paragraph vectors|
|sW|matrix[double]|weights (parameters) matrices for global statistics|
|sb|matrix[double]|biases vectors for global statistics|
|fW|matrix[double]|weights (parameters) matrices for  combining all trained features (final)|
|fb|matrix[double]|biases vectors for combining all trained features (final)|

## sherlockNet-Function


This function implements Neural Network for Sherlock: A Deep Learning Approach to Semantic Data Type Detection

Trains a 2 hidden layer softmax classifier.

REFERENCE:
### Usage


```python
sherlockNet(X_train, y_train, hidden_layer_neurons)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X_train|matrix[double]|---|input data matrix, of shape (N, D)|
|y_train|matrix[double]|---|target matrix, of shape (N, K)|
|hidden_layer_neurons|int|number|of neurons per hidden layer|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|W|matrix[double]|weights (parameters) matrix, of shape (D, M, 3).|
|b|matrix[double]|biases vector, of shape (1, M, 3).|

## sherlockPredict-Function


THis function implements prediction and evaluation phase of Sherlock:

Split feature matrix into four different feature categories and predicting the class probability

on the respective features. Then combine all predictions for final predicted probabilities.
### Usage


```python
sherlockPredict(X, cW, cb, wW, wb, pW, pb, sW, sb, fW, fb)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|matrix[double]|---|matrix of values which are to be classified|
|cW|matrix[double]|---|weights (parameters) matrices for character distribtions|
|cb|matrix[double]|---|biases vectors for character distribtions|
|wW|matrix[double]|---|weights (parameters) matrices for word embeddings|
|wb|matrix[double]|---|biases vectors for word embeddings|
|pW|matrix[double]|---|weights (parameters) matrices for paragraph vectors|
|pb|matrix[double]|---|biases vectors for paragraph vectors|
|sW|matrix[double]|---|weights (parameters) matrices for global statistics|
|sb|matrix[double]|---|biases vectors for global statistics|
|fW|matrix[double]|---|weights (parameters) matrices for  combining all trained features (final)|
|fb|matrix[double]|---|biases vectors for combining all trained features (final)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|probs|matrix[double]|class probabilities of shape (N, K)|

## shortestPath-Function


Computes the minimum distances (shortest-path) between a single source vertex and every other vertex in the graph.
### Usage


```python
shortestPath(G, (G, The, are, of, maxi, max, sourceNode, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|G|Matrix[Double]|---|adjacency matrix of the labeled graph: Such graph can be directed|
|(G|is|symmetric)|or undirected (G is not symmetric).|
|The|values|of|G can be 0/1 (just specifying whether the nodes|
|are|connected|or|not) or integer values (representing the weight|
|of|the|edges|or the distances between nodes, 0 if not connected).|
|maxi|Integer|0|Integer max number of iterations accepted (0 for FALSE, i.e.|
|max|number|of|iterations not defined)|
|sourceNode|Integer|node|index to calculate the shortest paths to all other nodes.|
|verbose|Boolean|FALSE|flag for verbose debug output|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|C|Matrix[Double]|Output matrix (double) of minimum distances (shortest-path) between|
|vertices:|The|value of the ith row and the jth column of the output|
|matrix|is|the minimum distance shortest-path from vertex i to vertex j.|
|When|the|value of the minimum distance is infinity, the two nodes are|
|not|connected.||

## slicefinder-Function


This builtin function imlements SliceLine, a linear-algebra-based

ML model debugging technique for finding the top-k data slices where
### Usage


```python
slicefinder(X, e, k, maxL, minSup, alpha, tpEval, tpBlksz, selFeat, the, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Recoded dataset into Matrix|
|e|Matrix[Double]|---|Trained model|
|k|Integer|1|Number of subsets required|
|maxL|Integer|maximum|level L (conjunctions of L predicates), 0 unlimited|
|minSup|Integer|minimum|support (min number of rows per slice)|
|alpha|Double|weight|[0,1]: 0 only size, 1 only error|
|tpEval|Boolean|flag|for task-parallel slice evaluation,|
|tpBlksz|Integer|block|size for task-parallel execution (num slices)|
|selFeat|Boolean|flag|for removing one-hot-encoded features that don't satisfy|
|the|initial|minimum-support|constraint and/or have zero error|
|verbose|Boolean|flag|for verbose debug output|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|TK|Matrix[Double]|top-k slices (k x ncol(X) if successful)|
|TKC|Matrix[Double]|score, size, error of slices (k x 3)|
|D|Matrix[Double]|debug matrix, populated with enumeration stats if verbose|

## smote-Function


Builtin function for handing class imbalance using Synthetic Minority Over-sampling Technique (SMOTE)

by Nitesh V. Chawla et. al. In Journal of Artificial Intelligence Research 16 (2002). 321â€“357
### Usage


```python
smote(X, mask, s, k)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of minority class samples|
|mask|Matrix[Double]|---|0/1 mask vector where 0 represent numeric value and 1 represent categorical value|
|s|Integer|25|Amount of SMOTE (percentage of oversampling), integral multiple of 100|
|k|Integer|1|Number of nearest neighbour|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Matrix of (N/100)-1 * nrow(X) synthetic minority class samples|

## softmax-Function


This is a softmax classifier,forward function Computes the forward pass for a softmax classifier.
### Usage


```python
softmax(S)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|S|matrix[double]|---|Inputs of shape (N, D).|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|P|matrix[double]|Outputs of shape (N, D).|

## split-Function


This function split input data X and Y into contiguous or samples train/test sets
### Usage


```python
split(X, Y, f, cont, seed)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|Y|Matrix[Double]|---|Input Labels|
|f|Double|0.7|Train set fraction [0,1]|
|cont|Boolean|TRUE|contiuous splits, otherwise sampled|
|seed|Integer|-1|The seed to reandomly select rows in sampled mode|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Xtrain|Matrix[Double]|Train split of feature matrix|
|Xtest|Matrix[Double]|Test split of feature matrix|
|ytrain|Matrix[Double]|Train split of label matrix|
|ytest|Matrix[Double]|Test split of label matrix|

## splitBalanced-Function


This functions split input data X and Y into contiguous balanced ratio

Related to [SYSTEMDS-2902] dependency function for cleaning pipelines
### Usage


```python
splitBalanced(X, Y, f, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|Y|Matrix[Double]|---|Input Labels|
|f|Double|0.7|Train set fraction [0,1]|
|verbose|Boolean|FALSE|print available|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X_train|Matrix[Double]|Train split of feature matrix|
|X_test|Matrix[Double]|Test split of feature matrix|
|y_train|Matrix[Double]|Train split of label matrix|
|y_test|Matrix[Double]|Test split of label matrix|

## stableMarriage-Function


This script computes a solution for the stable marriage problem.
### Usage


```python
stableMarriage(P, It, A, It, ordered, i.e., i.e., index)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|P|Matrix[Double]|---|proposer matrix P.|
|It|must|be|a square matrix with no zeros.|
|A|Matrix[Double]|---|acceptor matrix A.|
|It|must|be|a square matrix with no zeros.|
|ordered|Boolean|TRUE|If true, P and A are assumed to be ordered,|
|i.e.|the|leftmost|value in a row is the most preferred partner's index.|
|i.e.|the|leftmost|value in a row in P is the preference value for the acceptor with|
|index|1|and|vice-versa (higher is better).|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Matrix[Double]|Result|Matrix|
|If|cell|[i,j] is non-zero, it means that acceptor i has matched with|
|proposer|j.|Further, if cell [i,j] is non-zero, it holds the preference|
|value|that|led to the match.|

## statsNA-Function


The statsNA-function Print summary stats about the distribution of missing values in a univariate time series.
### Usage


```python
statsNA(X, bins, divided, missing, verbose, For, more)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Numeric Vector ('vector') object containing NAs|
|bins|Integer|4|Split number for bin stats. Number of bins the time series gets|
|divided|into.|For|each bin information about amount/percentage of|
|missing|values|is|printed.|
|verbose|Boolean|TRUE|Print detailed information.|
|For|print_only|=|TRUE, the missing value stats are printed with|
|more|information|("Stats|for Bins" and "overview NA series").|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|stats|Matrix[Double]|Double      Column vector where each row correspond to following,|
|1.|Length|of time series (including NAs)|
|2.|Number|of Missing Values (NAs)|
|3.|Percentage|of Missing Values (#2/#1)|
|4.|Number|of Gaps (consisting of one or more consecutive NAs)|
|5.|Average|Gap Size - Average size of consecutive NAs for the NA gaps|
|6.|Longest|NA gap - Longest series of consecutive missing values|
|7.|Most|frequent gap size - Most frequently occurring gap size|
|8.|Gap|size accounting for most NAs|

## stratstats-Function


The stratstats.dml script computes common bivariate statistics, such as correlation, slope, and their p-value,

in parallel for many pairs of input variables in the presence of a confounding categorical variable.
### Usage


```python
stratstats(X, Y, the, S, the, Xcid, the, Ycid, the, Scid)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix X that has all 1-st covariates|
|Y|Matrix[Double]|"|"     Matrix Y that has all 2-nd covariates|
|the|default|value|" " means "use X in place of Y"|
|S|Matrix[Double]|"|"     Matrix S that has the stratum column|
|the|default|value|" " means "use X in place of S"|
|Xcid|Matrix[Double]|"|"     1-st covariate X-column indices|
|the|default|value|" " means "use columns 1 : ncol(X)"|
|Ycid|Matrix[Double]|"|"     2-nd covariate Y-column indices|
|the|default|value|" " means "use columns 1 : ncol(Y)"|
|Scid|Int|1|Column index of the stratum column in S|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|OutMtx|Matrix[Double]|Output matrix, one row per each distinct pair|
|(1st|covariante,|2nd covariante)|
|40|columns|containing the following information:|
|Col|01:|1st covariate X-column number|
|Col|02:|1st covariate global presence count|
|Col|03:|1st covariate global mean|
|Col|04:|1st covariate global standard deviation|
|Col|05:|1st covariate stratified standard deviation|
|Col|06:|R-squared, 1st covariate vs. strata|
|Col|07:|adjusted R-squared, 1st covariate vs. strata|
|Col|08:|P-value, 1st covariate vs. strata|
|Col|09-10:|Reserved|
|Col|11:|2nd covariate Y-column number|
|Col|12:|2nd covariate global presence count|
|Col|13:|2nd covariate global mean|
|Col|14:|2nd covariate global standard deviation|
|Col|15:|2nd covariate stratified standard deviation|
|Col|16:|R-squared, 2nd covariate vs. strata|
|Col|17:|adjusted R-squared, 2nd covariate vs. strata|
|Col|18:|P-value, 2nd covariate vs. strata|
|Col|19-20:|Reserved|
|Col|21:|Global 1st & 2nd covariate presence count|
|Col|22:|Global regression slope (2nd vs. 1st covariate)|
|Col|23:|Global regression slope standard deviation|
|Col|24:|Global correlation = +/- sqrt(R-squared)|
|Col|25:|Global residual standard deviation|
|Col|26:|Global R-squared|
|Col|27:|Global adjusted R-squared|
|Col|28:|Global P-value for hypothesis "slope = 0"|
|Col|29-30:|Reserved|
|Col|31:|Stratified 1st & 2nd covariate presence count|
|Col|32:|Stratified regression slope (2nd vs. 1st covariate)|
|Col|33:|Stratified regression slope standard deviation|
|Col|34:|Stratified correlation = +/- sqrt(R-squared)|
|Col|35:|Stratified residual standard deviation|
|Col|36:|Stratified R-squared|
|Col|37:|Stratified adjusted R-squared|
|Col|38:|Stratified P-value for hypothesis "slope = 0"|
|Col|39:|Number of strata with at least two counted points|
|Col|40:|Reserved|

## tomeklink-Function


The tomekLink-function performs undersampling by removing Tomek's links for imbalanced multiclass problems

Computes TOMEK links and drops them from data matrix and label vector.

Drops only the majarity label and corresponding point of TOMEK links.
### Usage


```python
tomeklink(X, y)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix (nxm)|
|y|Matrix[Double]|---|Label Matrix (nx1), greater than zero|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|X_under|Matrix[Double]|Data Matrix without Tomek links|
|y_under|Matrix[Double]|Labels corresponding to undersampled data|
|drop_idx|Matrix[Double]|Indices of dropped rows/labels wrt input|

## toOneHot-Function


The toOneHot-function encodes unordered categorical vector to multiple binarized vectors.
### Usage


```python
toOneHot(X, numclasses)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|matrix[double]|---|Vector with N integer entries between 1 and numClasses|
|numclasses|int|---|Number of columns, must be be greater than or equal to largest value in X|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|matrix[double]|One-hot-encoded matrix with shape (N, numClasses)|

## topk_cleaning-Function


This function cleans top-K item (where K is given as input)for a given list of users.

metaData[3, ncol(X)] : metaData[1] stores mask, metaData[2] stores schema, metaData[3] stores FD mask
### Usage


```python
topk_cleaning(dataTrain, dataTest, metaData, primitives, topK, resource_val, sample, cv, cvk, isLastLabel, correctTypos)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|dataTrain|Frame[Unknown]|---||
|dataTest|Frame[Unknown]|NULL||
|metaData|Frame[Unknown]|NULL||
|primitives|Frame[Unknown]|---||
|topK|Integer|5||
|resource_val|Integer|20||
|sample|Double|0.1||
|cv|Boolean|TRUE||
|cvk|Integer|2||
|isLastLabel|Boolean|TRUE||
|correctTypos|Boolean|FALSE||

### Returns

|OUTPUT:|
| :---: |
|NAME            TYPE              MEANING|

## tSNE-Function


This function performs dimensionality reduction using tSNE algorithm based on

the paper: Visualizing Data using t-SNE, Maaten et. al.
### Usage


```python
tSNE(X, (number, reduced_dims, perplexity, lr, momentum, max_iter, seed, If, is_verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Data Matrix of shape|
|(number|of|data|points, input dimensionality)|
|reduced_dims|Integer|2|Output dimensionality|
|perplexity|Integer|30|Perplexity Parameter|
|lr|Double|300.|Learning rate|
|momentum|Double|0.9|Momentum Parameter|
|max_iter|Integer|1000|Number of iterations|
|seed|Integer|-1|The seed used for initial values.|
|If|set|to|-1 random seeds are selected.|
|is_verbose|Boolean|FALSE|Print debug information|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Data Matrix of shape (number of data points, reduced_dims)|

## univar-Function


Computes univariate statistics for all attributes in a given data set
### Usage


```python
univar(X, TYPES, 1)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input matrix of the shape (N, D)|
|TYPES|Matrix[Integer]|---|Matrix of the shape (1, D) with features types:|
|1|for|scale,|2 for nominal, 3 for ordinal|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|univarStats|Matrix[Double]|univariate statistics for all attributes|

## vectorToCsv-Function


This builtin function  convert vector into csv string such as [1 0 0 1 1 0 1] = "1,4,5,7"

Related to [SYSTEMDS-2662] dependency function for cleaning pipelines
### Usage


```python
vectorToCsv(mask)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|mask|Matrix[Double]|---|Data vector (having 0 for excluded indexes)|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|indexes|String|indexes|

## winsorize-Function


The winsorize-function removes outliers from the data. It does so by computing upper and

lower quartile range of the given data then it replaces any value that falls outside this range
### Usage


```python
winsorize(X, verbose)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Input feature matrix|
|verbose|Boolean|FALSE|To print output on screen|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|Matrix without outlier values|

## xdummy1-Function


This builtin function does the process of coding a categorical variable into dichotomous variables
### Usage


```python
xdummy1(X)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---||

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]||

## xdummy2-Function


This builtin function does the process of coding a categorical variable into dichotomous variables
### Usage


```python
xdummy2(X)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|---|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|Y|Matrix[Double]|---|

## xgboost-Function


XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting. This xgboost

implementation supports classification and regression and is capable of working with categorical and scalar features.
### Usage


```python
xgboost(X, Y, R, -, -, Feature, If, sml_type, num_trees, learning_rate, max_depth, lambda)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Feature matrix X; note that X needs to be both recoded and dummy coded|
|Y|Matrix[Double]|---|Label matrix Y; note that Y needs to be both recoded and dummy coded|
|R|Matrix[Double]|1,|1xn      Matrix R; 1xn vector which for each feature in X contains the following information|
|-|R[,1]:|1|(scalar feature)|
|-|R[,2]:|2|(categorical feature)|
|Feature|1|is|a scalar feature and features 2 is a categorical feature|
|If|R|is|not provided by default all variables are assumed to be scale (1)|
|sml_type|Integer|1|Supervised machine learning type: 1 = Regression(default), 2 = Classification|
|num_trees|Integer|7|Number of trees to be created in the xgboost model|
|learning_rate|Double|0.3|Alias: eta. After each boosting step the learning rate controls the weights of the new predictions|
|max_depth|Integer|6|Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit|
|lambda|Double|0.0|L2 regularization term on weights. Increasing this value will make model more conservative and reduce amount of leaves of a tree|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|M|Matrix[Double]|Matrix M where each column corresponds to a node in the learned tree|
|(the|first|node is the init prediction) and each row contains|
|the|following|information:|
|M[1,j]:|id|of node j (in a complete binary tree)|
|M[2,j]:|tree|id to which node j belongs|
|M[3,j]:|Offset|(no. of columns) to left child of j if j is an internal node, otherwise 0|
|M[4,j]:|Feature|index of the feature (scale feature id if the feature is|
|scale|or|categorical feature id if the feature is categorical)|
|that|node|j looks at if j is an internal node, otherwise 0|
|M[5,j]:|Type|of the feature that node j looks at if j is an internal node.|
|if|leaf|= 0, if scalar = 1, if categorical = 2|
|M[6:,j]:|If|j is an internal node: Threshold the example's feature value is|
|compared|to|is stored at M[6,j] if the feature chosen for j is scale,|
|otherwise|if|the feature chosen for j is categorical rows 6,7,... depict|
|the|value|subset chosen for j|
|If|j|is a leaf node 1 if j is impure and the number of samples at j > threshold, otherwise 0|

## xgboostPredictClassification-Function


XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting. This xgboost

implementation supports classification  and is capable of working with categorical features.
### Usage


```python
xgboostPredictClassification(X, M, learning_rate)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors we want to predict (X_test)|
|M|Matrix[Double]|---|The model created at xgboost|
|learning_rate|Matrix[Double]|0.3|The learning rate used in the model|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|P|Matrix[Double]|The predictions of the samples using the given xgboost model. (y_prediction)|

## xgboostPredictRegression-Function


XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting. This xgboost

implementation supports regression.
### Usage


```python
xgboostPredictRegression(X, M, learning_rate)
```
### Arguments

|NAME|TYPE|DEFAULT|MEANING|
| :---: | :---: | :---: | :---: |
|X|Matrix[Double]|---|Matrix of feature vectors we want to predict (X_test)|
|M|Matrix[Double]|---|The model created at xgboost|
|learning_rate|Matrix[Double]|0.3|The learning rate used in the model|

### Returns

|NAME|TYPE|MEANING|
| :---: | :---: | :---: |
|P|Matrix[Double]|The predictions of the samples using the given xgboost model. (y_prediction)|
