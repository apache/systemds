---
layout: global
title: SystemML Algorithms Reference - KNN
displayTitle: <a href="algorithms-reference.html">SystemML Algorithms Reference</a>
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->
#K Nearest Neighbor
##Description
Nearest neighbor analysis is a simple machine learning algorithm which solves the following search problem: for a given case, find the k most similar cases among the training points in a database. Additionally, this algorithm can be used for classification and regression problems as follows. Assume we are given a training dataset with known target values, and some test cases with unknown target values. To predict the target value for a given query case , we first find the k nearest neighbors of  in the dataset according to a distance metric, e.g., Euclidean distance or city block distance. Next we compute the target value for  based on the target values of its k nearest neighbors: we use the mean or median for regression problems and a majority voting for classification problems.
## Usage
    hadoop jar SystemML.jar -f knn.dml
                            -nvargs X=<file>
                                    T=<file>
                                    Y=[file]
                                    Y_T=[file]
                                    NNR=<file>
                                    PR=[file]
                                    K=[file]
                                    Select_k=[0|1]
                                    k_min=[int]
                                    k_max=[int]
                                    interval=[int]
                                    select_feature=[0|1]
                                    START_SELECTED=[file]
                                    feature_max=[int]
                                    SELECTED_FEATURE=[file]
                                    feature_importance=[0|1]
                                    FEATURE_IMPORTANCE_VALUE=[file]
                                    trans_continuous=[0|1]
                                    predict_con_tg=[0|1]
                                    k_values=[int]
                                    fmt=[format]
#### Arguments
* X:   Location to read the input feature matrix
* T:   Location to read the input matrix for nearest neighbor search
* Y:   Location to read the input matrix containing target values
* Y_T: Location to read the type of matrix Y: continuous (=1) or categorical (=2)
* K:   Location to write the result of the selected k value
* PR:  Location to write the predicted values. Note that this argument is optional when Y is NOT specified.
* NNR: Location to write the indices of nearest neighbors for T
* trans_continuous:    (default: 0) Option flag for continuous features transformed to [-1,1]: 0 = do not transform continuous variables; 1 = transform continuous variables
* select_k:    (default: 0) Use k selection algorithm to estimate k: 0 = do not perform k selection, 1 = perform k selection
* k_min:   (default: 1) Min k value. Note that this argument is available if select_k = 1
* k_max:   (default: 100) Max k value. Note that this argument is available if select_k = 1
* select_feature:  (default: 0) Use feature selection algorithm to select features: 0 = do not perform feature selection, 1 = perform feature selection
* START_SELECTED:  Location to read the indices of the initial features used in the feature selection process
* FEATURE_SELECTED:    Location to write the result of feature selection
* feature_max: (default: 10) Max number of features used for feature selection
* interval:    (default: 1000) Interval value for K selection. Note that this argument is available if select_k = 1.
* feature_importance:  (default: 0) Compute feature importance: 0 = do not compute feature importance, 1 = compute feature importance
* FEATURE_IMPORTANCE_VALUE:    Location to write the result of the computed feature importance. Note that this argument is available if feature_importance = 1.
* k_value: (default: 5) Parameter k for KNN. Note that this argument is ignored if select_k = 1.
* predict_con_tg:  (default: 0) Continuous target predict function: mean(=0) or median(=1)
* fmt: (default: “text”) Matrix output format for K, PR, NNR, FEATURE_SELECTED, and FEATURE_IMPORTANCE_VALUE, usually "text" or "csv"

####Examples
    hadoop jar SystemML.jar -f knn.dml
                            -nvargs X=/user/ml/train.mtx
                                    T=/user/ml/test.mtx
                                    Y=/user/ml/cl.csv
                                    Y_T=/user/ml/cl_types.csv
                                    NNR=/user/ml/NNR.csv
                                    PR=/user/ml/PR.csv
                                    K=/uer/ml/K_values.csv
                                    Select_k=1
                                    k_min=1
                                    k_max=100
                                    interval=1000
                                    select_feature=1
                                    START_SELECTED=/user/ml/start_selected.csv
                                    feature_max=5
                                    SELECTED_FEATURE=/user/ml/selected_features.csv
                                    feature_importance=1
                                    FEATURE_IMPORTANCE_VALUE=/user/ml/FIV.csv
                                    trans_continuous=1
                                    predict_con_tg=1
                                    k_values=5
                                    fmt=csv
## Details
####1. Distance Metrics
To compute the distance between two cases, this script supports the Euclidean distance and the city block distance. Given two cases, $ j:( x_{1,j}, x_{2,j}, \ldots, x_{3,j} ) $ and  $ k:( x_{1,k}, x_{2,k}, \ldots, x_{3,k} ) $ ,

the Euclidean distance between j and k is computed as  

$$
dist(j,k) = \sqrt{\sum_{i=1}^d\left( x_{i,j} - x_{i,j} \right)^2}
$$


and the city block distance is given by
$$
dist(j,k) = \sum_{i=1}^{d}\left|x_{i,j} - x_{i,j} \right|
$$


####2. Data preparation

To account for difference in scale, if required, continue features will be transformed to [-1,1] using the formula
$$
x^{'}_i = { { 2\left(x_i - min(x_i) \right) } \over { max(x_i) - min(x_i) } }
$$


where $min(x_i)$ and  $max(x_i)$ are the minimum and maximum value of continuous feature $x_i$ in training dataset, respectively.
For categorical features, the features are assumed to be dummy-coded.
####3. Selecting k
In order to efficiently determine the best number of neighbors k (the one with the least prediction error in terms of the average error rate for regression or sum-of-square error for classification problems) we follow similar approaches as proposed by Hamerly and Speegle (2010) and Moore and Lee (1994). In the following, we first describe the Algorithm of  Hamerly and Speegle (referred to as Optimized LOOCV) for finding best k. Next we discuss how to incorporate the optimizations proposed by Moore and Lee (1994) (referred to as the BRACE algorithm) to speed up the process. Finally we review Optimized LOOCV and BRACE combined for selecting best k.     

#####3.1  Optimized LOOCV
The main idea of the optimized Leave-One-Out cross validation (LOOCV) is to first find kmax nearest neighbors at once (since the (k+1)-nearest neighbors of a query point q already contains the k-nearest neighbors of q) and then use cross validation to search for the best , where  is the minimum reasonable value of k and  denote the maximum reasonable value of k  specified by user. Let n denote the number of training examples. The algorithm can be summarized as follows:
* Step 1. Initialize  $ error(k)=0$  for al l  $k  \in [k_{min},k_{max}] $


* Step 2. For all $ i  \in [i,n] $  compute $n_{i}$ the k-nearest neighbors (kNN) of a query example from a dataset of size n-1 where the query example xi is being held out.  


* Step 3. Sort $N_i$ by decreasing distance from $x_i$.


* Step 4. For $ j \in [i,|N_i|] $ , i.e., from the nearest to the furthest example in $N_i$, incrementally compute the prediction error  $ error(k)$ .


* Step 5. Return optimal $k^* = \underset{k  \in [k_{min},k_{max}] }{\mathrm{argmin}}  error(k)$

Note that in the algorithm above the number of kNN computations (step 2) is equal to the number of training examples which can be high for large datasets. In order to reduce the number of kNN computations, we can utilize the BRACE algorithm discussed in the next section.     

#####3.2  BRACE
The BRACE algorithm uses two techniques called RACE and BLOCKING which can be used to reduce the computation cost of LOOCV. The main idea of the algorithm is to eliminate models (each k in the kNN computation corresponds to a model) that are unlikely to be the best model by the means of statistical tests (RACE). Furthermore only one model is chosen among two or more models that produce almost identical predictions (BLOCKING). The algorithm assumes that LOOCV errors are distributed normally throughout the dataset. Let $err_j$ denote the LOOCV error for each model $M_j$. The errors  $err_j$ have a priori unknown mean ${\mu}_j^*$ and unknown variance ${\sigma}_j^*$ and the goal is to find the model with the least ${\mu}_j^*$. Let  ${\widehat{\mu}}_j(t)$  and ${\widehat{\sigma}}_j^2(t)$ denote the sample mean and sample variance of model j’s LOOCV errors up to iteration t, i.e.,

$$
\widehat{\mu}_j(t) ={1 \over t}\sum_{i=1}^{t}err_j(i)
$$

$$
{\widehat{\sigma}}_j^2(t) ={1 \over t-1}\sum_{i=1}^{t}(err_j(i)-{{\widehat{\mu}}_j(t)})^2
$$



As more data points are being examined the uncertainty of the true error  decreases. Before running the BRACE algorithm, the data points are first shuffled. During its execution the algorithm estimates for each pair of models $M_j$ and $M_{j^{'}}$ the values
$$
h_{j{j^{'}}}^*=err_j^* -err_{j^{'}}^*
$$
, where $err_j^*$  is the true LOOCV error for model $M_j$ when considering all the examples. Moreover the algorithm uses Bayesian statistics to put a probability distribution on $h_{j{j^{'}}}^*$ for all $ j{j^{'}}$. To this end, we maintain the following statistics
$$
{\widehat{\mu}}_{jj^{'}}^{h}(t) ={1 \over t}\sum_{i=1}^{t}err_j(i)-err_{j^{'}}(i)
$$

$$
{\widehat{\sigma}}_{jj^{'}}^h(t) =\sqrt{{1 \over t-1}\sum_{i=1}^{t}[{(err_j(i)-{{\widehat{\mu}}_j(i)})-\widehat{\mu}}_{jj^{'}}^h ]^2}
$$

Model $M_j$ is eliminated if for some $j^{'}$ it holds that
$$
Pr(h_{jj^{'}}^*<{-\gamma})<\sigma
$$
Above $\sigma >0$ (default value is 0.001) is the hypothesis test error rate indicating the probability of making an error on eliminating a model and $\gamma>0$  (default value is 0.001) is the error indifference parameter that defines the range of error values within which two models are considered similar. Note that $h_{jj^{'}}^*$ has (an approximate) normal distribution with mean $\widehat{\mu}_{jj^{'}}^{h}(t)$  and standard deviation ${\widehat{\sigma}}_{jj^{'}}^h(t)$ and, in particular,  $Z_{jj^{'}}(t)={h_{jj^{'}}^*-\widehat{\mu}_{jj^{'}}^{h}(t) \over {{\widehat{\sigma}}_{jj^{'}}^h(t) }}$ has (an approximate) standard normal distribution. Model $M_j$ is eliminated if for some $j^{'}$ we have that
$$

Pr({Z_{jj^{'}}(t)<{-\gamma-\widehat{\mu}_{jj^{'}}^{h}(t)  \over {\widehat{\sigma}}_{jj^{'}}^h(t) }})<\delta

$$

##### 3.3  Optimized LOOCV + BRACE

In the BRACE algorithm the hypothesis testing is applied after the examination of each example. For improved efficiency, similar to Hamerly and Speegle (2010), we introduce an interval t, as the number of examples examined before the hypothesis testing is performed. The final algorithm proceeds in the following steps:

* Step 1. For each $k  \in [k_{min},k_{max}] $ ( $k_{min}$ has default value 1,  $k_{max}$has default value 100) maintain a status active/inactive and three statistics on the examples processed so far:
  1. sum of prediction errors (SE):$\sum_{i=1}^terr_j(i)$
  2. sum of the squared prediction errors (SSE):$\sum_{i=1}^terr_j^2(i)$
  3. sum of error products (SEP):$\sum_{i=1}^terr_j(i)err_{j^{'}}(i)$


* Step 2. After each interval of t examples (default value is default value is$\sqrt{n}$ , where n is the number of training examples), compare each pair of active k models using the BRACE algorithm, i.e., when comparing two models  $j>j^{'}$if model j is unlikely to achieve lower prediction error than j’ or obtains similar prediction error as model j’ mark j as inactive.  

* Step 3. Stop if at least one of the conditions below holds:  

  - a)  only one k is active,  


  - b)  all example have been processed,  and output k with minimum LOOCV error. Otherwise go to step 1.

####4. Feature selection

In order to identify a small subset of features relevant for prediction we apply a combination of forward selection (FS) and the BRACE algorithm described above.

#####FS+BRACE
FS+BRACE is a wrapper approach that combines the well-known forward selection technique with the BRACE algorithm. Assume that the input is represented as a binary string $ S=( s_{1}s_{2}{\ldots}s_{3} ) $  , where $s_j$ is 1 if j-th feature ($j \in [1,d]$) is selected and 0 otherwise. The algorithm proceeds as follows:

* Step 1. Start with a predefined start string $S=S_{user}$ , where $S_{user}$ is a binary string that   represents a user-specified set of features to be included in the model ($S_{user=00 \ldots0}$ if no features are specified).
* Step 2. If the number of 1s in $S$ is more than or equal to $J$  (default value is 10 and corresponds to the number of additional features to be added to the model excluding the ones that are already selected by the user) or  only contains 1s, then stop and output $S$ . Otherwise use the BRACE algorithm to compare the current model with all models that have one more feature, i.e., to compare $S$ with all strings  that have the same bits as  set to 1 but with only one additional bit at position  set to 1 which is 0 in .   
* Step 3. If the winner of the race $S_j^*=S$ , then stop and output $S$. Otherwise $S=S_j^*$ and go to step 2.

#### 5. Combined k and feature selection
 In order to select  and identify relevant features at the same time we proceed as follows:

* Step 1. For each $k  \in [k_{min},k_{max}] $ ( $k_{min}$ has default value 1,  $k_{max}$has default value 100) in parallel select relevant features using FS+BRACE.
* Step 2. Output the combination of features and  with the smallest LOOCV error.

#### 6. Prediction
If a target variable is available, we can compute the predicted target value for a query case q based on its k nearest neighbors.

* Categorical target

  Suppose that the target variable of the training cases has $J$ categories $1,2\ldots,J$, . Themajority voting will be used to predict the target value of a query case  based on the target values of its nearest neighbors, i.e.
  $$
  \widehat{y}_{q} =arg\max_{i}\left\{\sum_{i \in Nearest(q)}I(y_i=j), j=1,2,\ldots,J\right\}
  $$


where ) is a set containing the indices of k nearest neighbor of , and  is an indicator function which takes value 1 if , and 0 if .

* Continuous target
  For a query case , mean or  median of the target values among its  nearest neighbors will be used to predict the target value of q.

#### 7. Feature importance
Suppose there are m  features$X_{(1)},X_{(2)},\ldots,X_{(m)}$   in the final model. Then the feature importance will be computed using the leave-one-out method described as follows:

* Step 1: $i = 1$

* Step 2: Delete the feature $X_{(i)}$ from the model, and build kNN model based on other m-1  features, compute the classification error rate if target is categorical or sum-of-squares error if the target is continuous, denote it as $e_{(i)}$.

* Step 3. If $i<m$, let $i=i+1$ , and go to step 2. Otherwise, go to step 4


  $$

  FI_{(i)} = {e_{(i)} + {1 \over m} \over \sum_{j=1}^m e_{(j) + 1} }, i=1,2,\ldots,m

  $$


## Returns
The script outputs the indices of the k nearest neighbours of the query points. Additionally, if requested, the following returns will be provided: predicted values, selected k values, indices of the selected features, and feature importance.

##Reference
* Arya and Mount (1993). Algorithms for fast vector quantization.  In Proceedings of the Data Compression Conference, pages 381--390.
* Andoni, A. and Indyk,P. (2008). Near-optimal hashing algorithm for approximate nearest neighbor in high dimensions. Communications of the ACM, 117-122.
* Hamerly G. and Speegle G. (2010) Efficient Model Selection for Large-Scale Nearest-Neighbor Data Mining. BNCOD, 6121 of Lecture Notes in Computer Science, 37-54.
* Li Y. and Lu B.-L.. (2009) Feature selection based on loss-margin of nearest neighbor classification. Pattern Recognition 42(9), 1914-1921.
* Moore A. and Lee M. S. (1994) Efficient Algorithms for Minimizing Cross Validation Error. In International Conference on Machine Learning. 190-198.
