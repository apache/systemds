---
layout: global
title: SystemML Algorithms Reference - Clustering
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

# 3. Clustering


## 3.1. K-Means Clustering

### Description

Given a collection of $n$ records with a pairwise similarity measure,
the goal of clustering is to assign a category label to each record so
that similar records tend to get the same label. In contrast to
multinomial logistic regression, clustering is an *unsupervised*
learning problem with neither category assignments nor label
interpretations given in advance. In $k$-means clustering, the records
$x_1, x_2, \ldots, x_n$ are numerical feature vectors of $\dim x_i = m$
with the squared Euclidean distance $\|x_i - x_{i'}\|_2^2$ as the
similarity measure. We want to partition $\\{x_1, \ldots, x_n\\}$ into $k$
clusters $\\{S_1, \ldots, S_k\\}$ so that the aggregated squared distance
from records to their cluster means is minimized:

$$
\begin{equation}
\textrm{WCSS}\,\,=\,\, \sum_{i=1}^n \,\big\|x_i - mean(S_j: x_i\in S_j)\big\|_2^2 \,\,\to\,\,\min
\end{equation}
$$ 

The aggregated distance measure in (1) is
called the *within-cluster sum of squares* (WCSS). It can be viewed as a
measure of residual variance that remains in the data after the
clustering assignment, conceptually similar to the residual sum of
squares (RSS) in linear regression. However, unlike for the RSS, the
minimization of (1) is an NP-hard
problem [[AloiseDHP2009]](algorithms-bibliography.html).

Rather than searching for the global optimum in (1), a
heuristic algorithm called Lloyd’s algorithm is typically used. This
iterative algorithm maintains and updates a set of $k$ *centroids*
$\\{c_1, \ldots, c_k\\}$, one centroid per cluster. It defines each
cluster $S_j$ as the set of all records closer to $c_j$ than to any
other centroid. Each iteration of the algorithm reduces the WCSS in two
steps:

  1. Assign each record to the closest centroid, making
$mean(S_j)\neq c_j$
  2. Reset each centroid to its cluster’s mean:
$c_j := mean(S_j)$

After Step 1, the centroids are generally
different from the cluster means, so we can compute another
"within-cluster sum of squares" based on the centroids:

$$\textrm{WCSS_C}\,\,=\,\, \sum_{i=1}^n \,\big\|x_i - \mathop{\textrm{centroid}}(S_j: x_i\in S_j)\big\|_2^2
\label{eqn:WCSS:C}$$ 

This WCSS\_C after Step 1
is less than the means-based WCSS before Step 1
(or equal if convergence achieved), and in Step 2
the WCSS cannot exceed the WCSS\_C for *the same* clustering; hence the
WCSS reduction.

Exact convergence is reached when each record becomes closer to its
cluster’s mean than to any other cluster’s mean, so there are no more
re-assignments and the centroids coincide with the means. In practice,
iterations may be stopped when the reduction in WCSS (or in WCSS\_C)
falls below a minimum threshold, or upon reaching the maximum number of
iterations. The initialization of the centroids is also an important
part of the algorithm. The smallest WCSS obtained by the algorithm is
not the global minimum and varies depending on the initial centroids. We
implement multiple parallel runs with different initial centroids and
report the best result.

**Scoring.** Our scoring script evaluates the clustering output by comparing it with
a known category assignment. Since cluster labels have no prior
correspondence to the categories, we cannot count "correct" and "wrong"
cluster assignments. Instead, we quantify them in two ways:

  1. Count how many same-category and different-category pairs of records end
up in the same cluster or in different clusters;
  2. For each category, count the prevalence of its most common cluster; for
each cluster, count the prevalence of its most common category.

The number of categories and the number of clusters ($k$) do not have to
be equal. A same-category pair of records clustered into the same
cluster is viewed as a "true positive," a different-category pair
clustered together is a "false positive," a same-category pair clustered
apart is a "false negative" etc.


### Usage

**K-Means**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans.dml
                            -nvargs X=<file>
                                    C=[file]
                                    k=<int>
                                    runs=[int]
                                    maxi=[int]
                                    tol=[double]
                                    samp=[int]
                                    isY=[boolean]
                                    Y=[file]
                                    fmt=[format]
                                    verb=[boolean]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=<file>
                                         C=[file]
                                         k=<int>
                                         runs=[int]
                                         maxi=[int]
                                         tol=[double]
                                         samp=[int]
                                         isY=[boolean]
                                         Y=[file]
                                         fmt=[format]
                                         verb=[boolean]
</div>
</div>

**K-Means Prediction**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans-predict.dml
                            -nvargs X=[file]
                                    C=[file]
                                    spY=[file]
                                    prY=[file]
                                    fmt=[format]
                                    O=[file]
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans-predict.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=[file]
                                         C=[file]
                                         spY=[file]
                                         prY=[file]
                                         fmt=[format]
                                         O=[file]
</div>
</div>


### Arguments - K-Means

**X**: Location to read matrix $X$ with the input data records as rows

**C**: (default: `"C.mtx"`) Location to store the output matrix with the best available
cluster centroids as rows

**k**: Number of clusters (and centroids)

**runs**: (default: `10`) Number of parallel runs, each run with different initial
centroids

**maxi**: (default: `1000`) Maximum number of iterations per run

**tol**: (default: `0.000001`) Tolerance (epsilon) for single-iteration WCSS\_C change ratio

**samp**: (default: `50`) Average number of records per centroid in data samples used
in the centroid initialization procedure

**Y**: (default: `"Y.mtx"`) Location to store the one-column matrix $Y$ with the best
available mapping of records to clusters (defined by the output
centroids)

**isY**: (default: `FALSE`) Do not write matrix $Y$

**fmt**: (default: `"text"`) Matrix file output format, such as `text`,
`mm`, or `csv`; see read/write functions in
SystemML Language Reference for details.

**verb**: (default: `FALSE`) Do not print per-iteration statistics for
each run


### Arguments - K-Means Prediction

**X**: (default: `" "`) Location to read matrix $X$ with the input data records as
rows, optional when `prY` input is provided

**C**: (default: `" "`) Location to read matrix $C$ with cluster centroids as rows,
optional when `prY` input is provided; NOTE: if both
X and C are provided, `prY` is an
output, not input

**spY**: (default: `" "`) Location to read a one-column matrix with the externally
specified "true" assignment of records (rows) to categories, optional
for prediction without scoring

**prY**: (default: `" "`) Location to read (or write, if X and
C are present) a column-vector with the predicted
assignment of rows to clusters; NOTE: No prior correspondence is assumed
between the predicted cluster labels and the externally specified
categories

**fmt**: (default: `"text"`) Matrix file output format for `prY`, such as
`text`, `mm`, or `csv`; see read/write
functions in SystemML Language Reference for details.

**0**: (default: `" "`) Location to write the output statistics defined in
[**Table 6**](algorithms-clustering.html#table6), by default print them to the
standard output


### Examples

**K-Means**:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans.dml
                            -nvargs X=/user/ml/X.mtx
                                    k=5
                                    C=/user/ml/centroids.mtx
                                    fmt=csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         k=5
                                         C=/user/ml/centroids.mtx
                                         fmt=csv
</div>
</div>

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans.dml
                            -nvargs X=/user/ml/X.mtx
                                    k=5
                                    runs=100
                                    maxi=5000
                                    tol=0.00000001
                                    samp=20
                                    C=/user/ml/centroids.mtx
                                    isY=1
                                    Y=/user/ml/Yout.mtx
                                    verb=1
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         k=5
                                         runs=100
                                         maxi=5000
                                         tol=0.00000001
                                         samp=20
                                         C=/user/ml/centroids.mtx
                                         isY=1
                                         Y=/user/ml/Yout.mtx
                                         verb=1
</div>
</div>

**K-Means Prediction**:

To predict Y given X and C:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans-predict.dml
                            -nvargs X=/user/ml/X.mtx
                                    C=/user/ml/C.mtx
                                    prY=/user/ml/PredY.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans-predict.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         C=/user/ml/C.mtx
                                         prY=/user/ml/PredY.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

To compare "actual" labels `spY` with "predicted" labels
given X and C:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans-predict.dml
                            -nvargs X=/user/ml/X.mtx
                                    C=/user/ml/C.mtx
                                    spY=/user/ml/Y.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans-predict.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs X=/user/ml/X.mtx
                                         C=/user/ml/C.mtx
                                         spY=/user/ml/Y.mtx
                                         O=/user/ml/stats.csv
</div>
</div>

To compare "actual" labels `spY` with given "predicted"
labels prY:

<div class="codetabs">
<div data-lang="Hadoop" markdown="1">
    hadoop jar SystemML.jar -f Kmeans-predict.dml
                            -nvargs spY=/user/ml/Y.mtx
                                    prY=/user/ml/PredY.mtx
                                    O=/user/ml/stats.csv
</div>
<div data-lang="Spark" markdown="1">
    $SPARK_HOME/bin/spark-submit --master yarn-cluster
                                 --conf spark.driver.maxResultSize=0
                                 --conf spark.akka.frameSize=128
                                 SystemML.jar
                                 -f Kmeans-predict.dml
                                 -config SystemML-config.xml
                                 -exec hybrid_spark
                                 -nvargs spY=/user/ml/Y.mtx
                                         prY=/user/ml/PredY.mtx
                                         O=/user/ml/stats.csv
</div>
</div>


* * *

<a name="table6" />
**Table 6**: The O-file for Kmeans-predict provides the
output statistics in CSV format, one per line, in the following
format: (NAME, \[CID\], VALUE). Note: the 1st group statistics are
given if X input is available; the 2nd group statistics
are given if X and C inputs are available;
the 3rd and 4th group statistics are given if spY input
is available; only the 4th group statistics contain a nonempty CID
value; when present, CID contains either the specified category label
or the predicted cluster label.

<table>
  <thead>
    <tr>
      <th>Inputs Available</th>
      <th>Name</th>
      <th>CID</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center" rowspan="5">X</td>
      <td>TSS</td>
      <td>&#160;</td>
      <td>Total Sum of Squares (from the total mean)</td>
    </tr>
    <tr>
      <td>WCSS_M</td>
      <td>&#160;</td>
      <td>Within-Cluster Sum of Squares (means as centers)</td>
    </tr>
    <tr>
      <td>WCSS_M_PC</td>
      <td>&#160;</td>
      <td>Within-Cluster Sum of Squares (means), in % of TSS</td>
    </tr>
    <tr>
      <td>BCSS_M</td>
      <td>&#160;</td>
      <td>Between-Cluster Sum of Squares (means as centers)</td>
    </tr>
    <tr>
      <td>BCSS_M_PC</td>
      <td>&#160;</td>
      <td>Between-Cluster Sum of Squares (means), in % of TSS</td>
    </tr>
    <tr>
      <td style="text-align: center" rowspan="4">X and C</td>
      <td>WCSS_C</td>
      <td>&#160;</td>
      <td>Within-Cluster Sum of Squares (centroids as centers)</td>
    </tr>
    <tr>
      <td>WCSS_C_PC</td>
      <td>&#160;</td>
      <td>Within-Cluster Sum of Squares (centroids), % of TSS</td>
    </tr>
    <tr>
      <td>BCSS_C</td>
      <td>&#160;</td>
      <td>Between-Cluster Sum of Squares (centroids as centers)</td>
    </tr>
    <tr>
      <td>BCSS_C_PC</td>
      <td>&#160;</td>
      <td>Between-Cluster Sum of Squares (centroids), % of TSS</td>
    </tr>
    <tr>
      <td style="text-align: center" rowspan="8">spY</td>
      <td>TRUE_SAME_CT</td>
      <td>&#160;</td>
      <td>Same-category pairs predicted as Same-cluster, count</td>
    </tr>
    <tr>
      <td>TRUE_SAME_PC</td>
      <td>&#160;</td>
      <td>Same-category pairs predicted as Same-cluster, %</td>
    </tr>
    <tr>
      <td>TRUE_DIFF_CT</td>
      <td>&#160;</td>
      <td>Diff-category pairs predicted as Diff-cluster, count</td>
    </tr>
    <tr>
      <td>TRUE_DIFF_PC</td>
      <td>&#160;</td>
      <td>Diff-category pairs predicted as Diff-cluster, %</td>
    </tr>
    <tr>
      <td>FALSE_SAME_CT</td>
      <td>&#160;</td>
      <td>Diff-category pairs predicted as Same-cluster, count</td>
    </tr>
    <tr>
      <td>FALSE_SAME_PC</td>
      <td>&#160;</td>
      <td>Diff-category pairs predicted as Same-cluster, %</td>
    </tr>
    <tr>
      <td>FALSE_DIFF_CT</td>
      <td>&#160;</td>
      <td>Same-category pairs predicted as Diff-cluster, count</td>
    </tr>
    <tr>
      <td>FALSE_DIFF_PC</td>
      <td>&#160;</td>
      <td>Same-category pairs predicted as Diff-cluster, %</td>
    </tr>
    <tr>
      <td style="text-align: center" rowspan="8">spY</td>
      <td>SPEC_TO_PRED</td>
      <td style="text-align: center">+</td>
      <td>For specified category, the best predicted cluster id</td>
    </tr>
    <tr>
      <td>SPEC_FULL_CT</td>
      <td style="text-align: center">+</td>
      <td>For specified category, its full count</td>
    </tr>
    <tr>
      <td>SPEC_MATCH_CT</td>
      <td style="text-align: center">+</td>
      <td>For specified category, best-cluster matching count</td>
    </tr>
    <tr>
      <td>SPEC_MATCH_PC</td>
      <td style="text-align: center">+</td>
      <td>For specified category, % of matching to full count</td>
    </tr>
    <tr>
      <td>PRED_TO_SPEC</td>
      <td style="text-align: center">+</td>
      <td>For predicted cluster, the best specified category id</td>
    </tr>
    <tr>
      <td>PRED_FULL_CT</td>
      <td style="text-align: center">+</td>
      <td>For predicted cluster, its full count</td>
    </tr>
    <tr>
      <td>PRED_MATCH_CT</td>
      <td style="text-align: center">+</td>
      <td>For predicted cluster, best-category matching count</td>
    </tr>
    <tr>
      <td>PRED_MATCH_PC</td>
      <td style="text-align: center">+</td>
      <td>For predicted cluster, % of matching to full count</td>
    </tr>
  </tbody>
</table>

* * *


### Details

Our clustering script proceeds in 3 stages: centroid initialization,
parallel $k$-means iterations, and the best-available output generation.
Centroids are initialized at random from the input records (the rows
of $X$), biased towards being chosen far apart from each other. The
initialization method is based on the `k-means++` heuristic
from [[ArthurVassilvitskii2007]](algorithms-bibliography.html), with one important difference: to
reduce the number of passes through $X$, we take a small sample of $X$
and run the `k-means++` heuristic over this sample. Here is,
conceptually, our centroid initialization algorithm for one clustering
run:

  <ol>
    <li>Sample the rows of $X$ uniformly at random, picking each row with
probability $p = ks / n$ where
    <ul>
      <li>$k$ is the number of centroids</li>
      <li>$n$ is the number of records</li>
      <li>$s$ is the samp input parameter</li>
    </ul>
     If $ks \geq n$, the entire $X$ is used in place of its sample.
     </li>
     <li>Choose the first centroid uniformly at random from the sampled rows.</li>
     <li>Choose each subsequent centroid from the sampled rows, at random, with
probability proportional to the squared Euclidean distance between the
row and the nearest already-chosen centroid.</li>
  </ol>

The sampling of $X$ and the selection of centroids are performed
independently and in parallel for each run of the $k$-means algorithm.
When we sample the rows of $X$, rather than tossing a random coin for
each row, we compute the number of rows to skip until the next sampled
row as $\lceil \log(u) / \log(1 - p) \rceil$ where $u\in (0, 1)$ is
uniformly random. This time-saving trick works because

$$Prob[k-1 < \log_{1-p}(u) < k] \,\,=\,\, p(1-p)^{k-1} \,\,=\,\,
Prob[\textrm{skip $k-1$ rows}]$$

However, it requires us to estimate the maximum sample size, which we
set near $ks + 10\sqrt{ks}$ to make it generous enough.

Once we selected the initial centroid sets, we start the $k$-means
iterations independently in parallel for all clustering runs. The number
of clustering runs is given as the runs input parameter.
Each iteration of each clustering run performs the following steps:

  * Compute the centroid-dependent part of squared Euclidean distances from
all records (rows of $X$) to each of the $k$ centroids using matrix
product.
  * Take the minimum of the above for each record.
  * Update the current within-cluster sum of squares (WCSS) value, with
centroids substituted instead of the means for efficiency.
  * Check the convergence
criterion:
$$\textrm{WCSS}_{\mathrm{old}} - \textrm{WCSS}_{\mathrm{new}} < {\varepsilon}\cdot\textrm{WCSS}_{\mathrm{new}}$$
as
well as the number of iterations limit.
  * Find the closest centroid for each record, sharing equally any records
with multiple closest centroids.
  * Compute the number of records closest to each centroid, checking for
"runaway" centroids with no records left (in which case the run fails).
  * Compute the new centroids by averaging the records in their clusters.

When a termination condition is satisfied, we store the centroids and
the WCSS value and exit this run. A run has to satisfy the WCSS
convergence criterion to be considered successful. Upon the termination
of all runs, we select the smallest WCSS value among the successful
runs, and write out this run’s centroids. If requested, we also compute
the cluster assignment of all records in $X$, using integers from 1
to $k$ as the cluster labels. The scoring script can then be used to
compare the cluster assignment with an externally specified category
assignment.


### Returns

We output the $k$ centroids for the best available clustering,
i. e. whose WCSS is the smallest of all successful runs. The centroids
are written as the rows of the $k\,{\times}\,m$-matrix into the output
file whose path/name was provided as the `C` input
argument. If the input parameter `isY` was set
to `1`, we also output the one-column matrix with the cluster
assignment for all the records. This assignment is written into the file
whose path/name was provided as the `Y` input argument. The
best WCSS value, as well as some information about the performance of
the other runs, is printed during the script execution. The scoring
script `Kmeans-predict.dml` prints all its results in a
self-explanatory manner, as defined in
[**Table 6**](algorithms-clustering.html#table6).


