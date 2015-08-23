---
layout: global
title: SystemML Algorithms Reference - Matrix Factorization
displayTitle: <a href="algorithms-reference.html">SystemML Algorithms Reference</a>
---
      

# 5 Matrix Factorization


## 5.1 Principle Component Analysis

### Description

Principle Component Analysis (PCA) is a simple, non-parametric method to
transform the given data set with possibly correlated columns into a set
of linearly uncorrelated or orthogonal columns, called *principle
components*. The principle components are ordered in such a way
that the first component accounts for the largest possible variance,
followed by remaining principle components in the decreasing order of
the amount of variance captured from the data. PCA is often used as a
dimensionality reduction technique, where the original data is projected
or rotated onto a low-dimensional space with basis vectors defined by
top-$K$ (for a given value of $K$) principle components.


### Usage

    hadoop jar SystemML.jar -f PCA.dml
                            -nvargs INPUT=file 
                                    K=int
                                    CENTER=int
                                    SCALE=int
                                    PROJDATA=int 
                                    OFMT=format 
                                    MODEL=file 
                                    OUTPUT=file


#### Arguments

**INPUT**: Location (on HDFS) to read the input matrix.

**K**: Indicates dimension of the new vector space constructed from $K$
    principle components. It must be a value between $1$ and the number
    of columns in the input data.

**CENTER**: (default: 0) 0 or 1. Indicates whether or not to
    *center* input data prior to the computation of
    principle components.

**SCALE**: (default: 0) 0 or 1. Indicates whether or not to
    *scale* input data prior to the computation of
    principle components.

**PROJDATA**: 0 or 1. Indicates whether or not the input data must be projected
    on to new vector space defined over principle components.

**OFMT**: (default: `"csv"`) Specifies the output format.
    Choice of comma-separated values (`csv`) or as a sparse-matrix (`text`).

**MODEL**: Either the location (on HDFS) where the computed model is
    stored; or the location of an existing model.

**OUTPUT**: Location (on HDFS) to store the data rotated on to the new
    vector space.



#### Examples

    hadoop jar SystemML.jar -f PCA.dml 
                            -nvargs INPUT=/user/ml/input.mtx 
                            K=10
                            CENTER=1 
                            SCALE=1O
                            FMT=csv 
                            PROJDATA=1
                            OUTPUT=/user/ml/pca_output/

    hadoop jar SystemML.jar -f PCA.dml 
                            -nvargs INPUT=/user/ml/test_input.mtx 
                                    K=10 
                                    CENTER=1
                                    SCALE=1O 
                                    FMT=csv 
                                    PROJDATA=1
                                    MODEL=/user/ml/pca_output/ 
                                    OUTPUT=/user/ml/test_output.mtx  



#### Details

Principle Component Analysis (PCA) is a non-parametric procedure for
orthogonal linear transformation of the input data to a new coordinate
system, such that the greatest variance by some projection of the data
comes to lie on the first coordinate (called the first principal
component), the second greatest variance on the second coordinate, and
so on. In other words, PCA first selects a normalized direction in
$m$-dimensional space ($m$ is the number of columns in the input data)
along which the variance in input data is maximized – this is referred
to as the first principle component. It then repeatedly finds other
directions (principle components) in which the variance is maximized. At
every step, PCA restricts the search for only those directions that are
perpendicular to all previously selected directions. By doing so, PCA
aims to reduce the redundancy among input variables. To understand the
notion of redundancy, consider an extreme scenario with a data set
comprising of two variables, where the first one denotes some quantity
expressed in meters, and the other variable represents the same quantity
but in inches. Both these variables evidently capture redundant
information, and hence one of them can be removed. In a general
scenario, keeping solely the linear combination of input variables would
both express the data more concisely and reduce the number of variables.
This is why PCA is often used as a dimensionality reduction technique.

The specific method to compute such a new coordinate system is as
follows – compute a covariance matrix $C$ that measures the strength of
correlation among all pairs of variables in the input data; factorize
$C$ according to eigen decomposition to calculate its eigenvalues and
eigenvectors; and finally, order eigenvectors in the decreasing order of
their corresponding eigenvalue. The computed eigenvectors (also known as
*loadings*) define the new coordinate system and the square
root of eigen values provide the amount of variance in the input data
explained by each coordinate or eigenvector.


#### Returns

When MODEL is not provided, PCA procedure is
applied on INPUT data to generate MODEL as well as the rotated data
OUTPUT (if PROJDATA is set to $1$) in the new coordinate system. The
produced model consists of basis vectors MODEL$/dominant.eigen.vectors$
for the new coordinate system; eigen values
MODEL$/dominant.eigen.values$; and the standard deviation
MODEL$/dominant.eigen.standard.deviations$ of principle components. When
MODEL is provided, INPUT data is rotated according to the coordinate
system defined by MODEL$/dominant.eigen.vectors$. The resulting data is
stored at location OUTPUT.


* * *

